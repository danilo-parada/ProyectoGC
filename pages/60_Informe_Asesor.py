# pages/60_Informe_Asesor.py
from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, date

from lib_common import (
    get_df_norm, general_date_filters_ui, apply_general_filters,
    advanced_filters_ui, apply_advanced_filters, money, one_decimal, header_ui,
    style_table,
)
from lib_metrics import ensure_derived_fields, compute_kpis
from lib_report import excel_bytes_single, generate_pdf_report

# -------------------- Config & Header --------------------
st.set_page_config(page_title="Informe Asesor", layout="wide")
header_ui(
    title="Informe para la Toma de Decisiones Financieras",
    current_page="Informe Asesor",
    subtitle="KPIs, deuda y priorización con foco en cuentas especiales y proveedores prioritarios"
)

# -------------------- Estilos locales para tablas --------------------
TABLE_HEADER_BG = "#003399"
TABLE_HEADER_FG = "#FFFFFF"
TABLE_STRIPED_BG = "#f5f7ff"
TABLE_HOVER_BG = "#e8edff"
TABLE_FONT_SIZE = "15px"


def _table_style(df_disp: pd.DataFrame | pd.io.formats.style.Styler):
    """Aplica el estilo de tablas usado en Rankings para tablas estáticas."""

    if isinstance(df_disp, pd.DataFrame):
        sty = df_disp.style
    else:
        sty = df_disp

    sty = sty.hide(axis="index")
    sty = sty.set_table_styles([
        {"selector": "thead tr", "props": [
            ("background-color", TABLE_HEADER_BG),
            ("color", TABLE_HEADER_FG),
            ("font-weight", "bold"),
            ("font-size", TABLE_FONT_SIZE),
            ("text-align", "center"),
            ("border-radius", "12px 12px 0 0")
        ]},
        {"selector": "th", "props": [
            ("background-color", "transparent"),
            ("color", TABLE_HEADER_FG),
            ("font-weight", "600"),
            ("font-size", TABLE_FONT_SIZE),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.4px"),
            ("padding", "12px 18px"),
            ("text-align", "center")
        ]},
        {"selector": "tbody td", "props": [
            ("font-size", TABLE_FONT_SIZE),
            ("padding", "12px 18px"),
            ("text-align", "right"),
            ("border-bottom", "1px solid #e0e6ff")
        ]},
        {"selector": "tbody tr:nth-child(even)", "props": [
            ("background-color", TABLE_STRIPED_BG)
        ]},
        {"selector": "tbody tr:hover", "props": [
            ("background-color", TABLE_HOVER_BG)
        ]},
        {"selector": "tbody td:first-child", "props": [
            ("text-align", "left"),
            ("font-weight", "600"),
            ("color", "#1f2a55")
        ]},
    ], overwrite=False)
    return sty

# -------------------- Carga de base --------------------
df0 = get_df_norm()
if df0 is None:
    st.warning("Carga tus datos en 'Carga de Data' primero.")
    st.stop()

df0 = ensure_derived_fields(df0)

# -------------------- Filtros globales --------------------
fac_ini, fac_fin, pay_ini, pay_fin = general_date_filters_ui(df0)
# Avanzado sin prioritario global (usamos ese filtro en secciones locales)
sede, org, prov, cc, oc, est, _ = advanced_filters_ui(df0)

df = apply_general_filters(df0, fac_ini, fac_fin, pay_ini, pay_fin)
df = apply_advanced_filters(df, sede, org, prov, cc, oc, est, prio=[])

df_pag = df[df["estado_pago"] == "pagada"].copy()
df_nopag = df[df["estado_pago"] != "pagada"].copy()

TODAY = pd.Timestamp(date.today()).normalize()

# -------------------- Utils --------------------
def safe_export(df_in: pd.DataFrame, desired_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in desired_cols if c in df_in.columns]
    return df_in[cols].copy()

def ensure_dias_a_vencer(dfin: pd.DataFrame) -> pd.DataFrame:
    d = dfin.copy()
    if "dias_a_vencer" not in d.columns and "fecha_venc_30" in d:
        d["dias_a_vencer"] = (d["fecha_venc_30"] - TODAY).dt.days
    return d

def ensure_importe_deuda(dfin: pd.DataFrame) -> pd.DataFrame:
    d = dfin.copy()
    if "importe_deuda" not in d.columns:
        maut = pd.to_numeric(d.get("monto_autorizado"), errors="coerce").fillna(0.0)
        mfac = pd.to_numeric(d.get("fac_monto_total"), errors="coerce").fillna(0.0)
        est = d.get("estado_pago")
        d["importe_deuda"] = np.where(est.eq("autorizada_sin_pago"), maut, 0.0) + \
                             np.where(est.eq("sin_autorizacion"), mfac, 0.0)
    return d

def _fmt_count(n: int) -> str:
    return f"{n:,} doc."

def _agg_block(d: pd.DataFrame, mask):
    sub = d[mask].copy()
    monto = float(pd.to_numeric(sub.get("importe_deuda"), errors="coerce").fillna(0.0).sum())
    cant = int(len(sub))
    return monto, cant

def _card_html(
    title: str,
    value: str,
    subtitle: str | None = None,
    *,
    tag: str | None = None,
    tag_variant: str = "success",
    tone: str = "default",
    stats: list[tuple[str, str]] | None = None,
    compact: bool = True,
) -> str:
    classes = ["app-card", "app-card--frost"]
    if compact:
        classes.append("app-card__mini")
    if tone == "accent":
        classes.append("app-card--accent")
    title_html = html.escape(str(title))
    value_html = html.escape(str(value))
    subtitle_html = f'<p style="margin:0;color:var(--app-text-muted);font-size:0.85rem;">{html.escape(str(subtitle))}</p>' if subtitle else ""
    tag_html = ""
    if tag:
        tag_cls = "app-card__tag"
        if tag_variant == "warning":
            tag_cls += " app-card__tag--warning"
        tag_html = f'<span class="{tag_cls}">{html.escape(str(tag))}</span>'
    stats_html = ""
    if stats:
        pills = "".join(
            f'<div class="app-inline-stats__item"><span style="font-weight:600;color:var(--app-text);">{html.escape(str(label))}:</span> {html.escape(str(val))}</div>'
            for label, val in stats
        )
        stats_html = f'<div class="app-inline-stats">{pills}</div>'
    return (
        f'<div class="{" ".join(classes)}">'
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
    st.markdown(f'<div class="{wrapper}">{"".join(cards)}</div>', unsafe_allow_html=True)

def _fmt_days(val: float) -> str:
    return "s/d" if pd.isna(val) else f"{one_decimal(val)} d"

def _fmt_pct(val: float) -> str:
    return "s/d" if pd.isna(val) else f"{one_decimal(val)}%"

# =========================================================
# 1) KPIs
# =========================================================
st.subheader("1) Resumen KPIs de Pagos (Facturas Pagadas)")
if df.empty:
    st.info("No hay datos con los filtros actuales.")
else:
    kpi_total = compute_kpis(df)
    total_docs = len(df)
    total_pagadas = len(df_pag)
    cards = [
        _card_html(
            "Total facturado",
            money(kpi_total["total_facturado"]),
            subtitle="Base filtrada",
            tag=f"{total_docs:,} doc.",
            tone="accent",
        ),
        _card_html(
            "Total pagado (aut.)",
            money(kpi_total["total_pagado_aut"]),
            subtitle="Facturas con pago autorizado",
            tag=f"{total_pagadas:,} pagos",
        ),
        _card_html(
            "DSO promedio",
            _fmt_days(kpi_total["dso"]),
            subtitle="Emision -> pago",
        ),
        _card_html(
            "TFA promedio",
            _fmt_days(kpi_total["tfa"]),
            subtitle="Emision -> contabilizacion",
        ),
        _card_html(
            "TPA promedio",
            _fmt_days(kpi_total["tpa"]),
            subtitle="Contabilizacion -> pago",
        ),
        _card_html(
            "Gap %",
            _fmt_pct(kpi_total["gap_pct"]),
            subtitle="Brecha contable vs facturado",
            tag_variant="warning",
        ),
    ]
    _render_cards(cards)

    if "cuenta_especial" in df.columns:
        st.markdown('<div class="app-separator"></div>', unsafe_allow_html=True)
        st.markdown("### Desglose por cuenta especial")
        segment_cards: list[str] = []
        for flag in (True, False):
            sub = df[df["cuenta_especial"] == flag]
            k = compute_kpis(sub) if not sub.empty else compute_kpis(df.iloc[0:0])
            stats = [
                ("Total pagado", money(k["total_pagado_aut"])),
                ("DSO", _fmt_days(k["dso"])),
                ("TFA", _fmt_days(k["tfa"])),
                ("TPC", _fmt_days(k["tpa"])),
                ("Gap %", _fmt_pct(k["gap_pct"])),
            ]
            segment_cards.append(
                _card_html(
                    title=f"CE {'Si' if flag else 'No'}",
                    value=money(k["total_facturado"]),
                    subtitle=f"{len(sub):,} doc." if len(sub) else "0 doc.",
                    stats=stats,
                    compact=False,
                    tone="accent" if flag else "default",
                )
            )
        _render_cards(segment_cards, layout="grid-2")

# =========================================================
# 2) Top Proveedores
# =========================================================
st.subheader("2) Top 5 Proveedores (por Monto Contabilizado)")

def build_top_proveedores(df_in: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    cols_out = [
        "Razón Social","Monto Contabilizado","Monto Pagado","Días Promedio Pago",
        "Cant. Fact. ≤30 días","Cant. Fact. >30 días",
        "Proveedor Prioritario","Cuenta Especial"
    ]
    if df_in.empty:
        return pd.DataFrame(columns=cols_out)

    d = df_in.copy()
    d["dias_a_pago_calc"] = pd.to_numeric(d.get("dias_a_pago_calc"), errors="coerce")
    d["monto_autorizado"] = pd.to_numeric(d.get("monto_autorizado"), errors="coerce").fillna(0.0)
    d["monto_pagado"] = pd.to_numeric(d.get("monto_pagado"), errors="coerce").fillna(0.0)
    d["prov_prioritario"] = d.get("prov_prioritario", False).astype(bool)
    d["cuenta_especial"] = d.get("cuenta_especial", False).astype(bool)

    grp = (
        d.groupby("prr_razon_social", dropna=False)
         .agg(
             **{
                 "Monto Contabilizado": ("monto_autorizado", "sum"),
                 "Monto Pagado": ("monto_pagado", "sum"),
                 "Días Promedio Pago": ("dias_a_pago_calc", lambda s: s[s >= 0].mean()),
                 "Cant. Fact. ≤30 días": ("dias_a_pago_calc", lambda s: (s <= 30).sum()),
                 "Cant. Fact. >30 días": ("dias_a_pago_calc", lambda s: (s > 30).sum()),
                 "Proveedor Prioritario": ("prov_prioritario", "mean"),
                 "Cuenta Especial": ("cuenta_especial", "mean"),
             }
         )
         .reset_index()
         .rename(columns={"prr_razon_social": "Razón Social"})
         .sort_values("Monto Contabilizado", ascending=False)
         .head(top_n)
    )
    grp["Proveedor Prioritario"] = grp["Proveedor Prioritario"].apply(lambda v: "Sí" if v >= 0.5 else "No")
    grp["Cuenta Especial"] = grp["Cuenta Especial"].apply(lambda v: "Sí" if v >= 0.5 else "No")
    return grp[cols_out]

if not df_pag.empty:
    top_n = st.slider("Top N", 3, 20, 5, 1)
    rankings_df = build_top_proveedores(df_pag, top_n=top_n)
    rankings_display = rankings_df.assign(
        **{
            "Monto Contabilizado": rankings_df["Monto Contabilizado"].map(money),
            "Monto Pagado": rankings_df["Monto Pagado"].map(money),
            "Días Promedio Pago": rankings_df["Días Promedio Pago"].map(one_decimal),
        }
    )
    style_table(_table_style(rankings_display))
    st.download_button(
        "⬇️ Descargar Ranking",
        data=excel_bytes_single(rankings_df, "RankingProveedores"),
        file_name="ranking_proveedores.xlsx",
        disabled=rankings_df.empty
    )
else:
    st.info("No hay facturas pagadas para rankings.")

# =========================================================
# 3) Análisis de Deuda Pendiente
# =========================================================
st.subheader("3) Análisis de Deuda Pendiente")
df_nopag = ensure_dias_a_vencer(ensure_importe_deuda(df_nopag))

def _kpis_deuda(dfin: pd.DataFrame) -> dict:
    if dfin.empty or "dias_a_vencer" not in dfin:
        return dict(vencido=0.0,c_venc=0, hoy=0.0, c_hoy=0, por_ven=0.0,c_por=0)
    vencido, c_v = _agg_block(dfin, dfin["dias_a_vencer"] < 0)
    hoy_m, c_h = _agg_block(dfin, dfin["dias_a_vencer"] == 0)
    porv, c_p = _agg_block(dfin, dfin["dias_a_vencer"] > 0)
    return dict(vencido=vencido, c_venc=c_v, hoy=hoy_m, c_hoy=c_h, por_ven=porv, c_por=c_p)

if df_nopag.empty:
    st.info("No hay documentos pendientes.")
else:
    n1 = df_nopag[df_nopag["Nivel"].eq("Doc. Autorizado p/ Pago")] if "Nivel" in df_nopag else df_nopag.iloc[0:0]
    n2 = df_nopag[df_nopag["Nivel"].eq("Doc. Pendiente de Autorización")] if "Nivel" in df_nopag else df_nopag.iloc[0:0]

def draw_debt_panel(title: str, dpanel: pd.DataFrame):
    st.markdown('<div class="app-title-block"><h3>' + html.escape(title) + '</h3><p>Desglose por cuenta especial</p></div>', unsafe_allow_html=True)
    if "cuenta_especial" not in dpanel:
        st.info("Sin campo de Cuenta Especial para desglosar.")
        return
    cards: list[str] = []
    for flag in (True, False):
        sub = dpanel[dpanel["cuenta_especial"] == flag]
        kk = _kpis_deuda(sub)
        total = kk["vencido"] + kk["hoy"] + kk["por_ven"]
        stats = [
            ("Vencida", f"{money(kk['vencido'])} | {_fmt_count(kk['c_venc'])}"),
            ("Hoy", f"{money(kk['hoy'])} | {_fmt_count(kk['c_hoy'])}"),
            ("Por vencer", f"{money(kk['por_ven'])} | {_fmt_count(kk['c_por'])}"),
        ]
        cards.append(
            _card_html(
                title=f"CE {'Si' if flag else 'No'}",
                value=money(total),
                subtitle=f"{len(sub):,} doc." if len(sub) else "0 doc.",
                stats=stats,
                compact=False,
                tone="accent" if flag else "default",
            )
        )
    _render_cards(cards, layout="grid-2")

draw_debt_panel("Doc. Autorizado p/ Pago", n1)
draw_debt_panel("Doc. Pendiente de Autorizacion", n2)


# =========================================================
# 4) Proyección de Vencimientos y Tablas de Pago
# =========================================================
st.subheader("4) Proyección de Vencimientos y Tablas de Pago")
st.caption("Los filtros siguientes impactan esta sección y las tablas/presupuesto hacia abajo.")

colf1, colf2, colf3 = st.columns([1,1,2])
ce_local = colf1.radio("Cuenta Especial (local)", ["Todas","Cuenta Especial","No Cuenta Especial"], horizontal=True, index=0)
prio_local = colf2.radio("Proveedor Prioritario (local)", ["Todos","Prioritario","No Prioritario"], horizontal=True, index=0)
horizonte = colf3.number_input("Horizonte (días)", 7, 90, 30, 7)

crit_sel = st.radio("Criterio de Orden", ["Riesgo de aprobación", "Urgencia de vencimiento"], horizontal=True)

def _apply_local_filters(dfin: pd.DataFrame) -> pd.DataFrame:
    out = dfin.copy()
    if ce_local == "Cuenta Especial":
        out = out[out["cuenta_especial"] == True]
    elif ce_local == "No Cuenta Especial":
        out = out[out["cuenta_especial"] == False]
    if prio_local == "Prioritario":
        out = out[out["prov_prioritario"] == True]
    elif prio_local == "No Prioritario":
        out = out[out["prov_prioritario"] == False]
    return out

df_nopag_loc = _apply_local_filters(df_nopag)

# KPIs locales
vencidos_m, vencidos_c = _agg_block(df_nopag_loc, df_nopag_loc["dias_a_vencer"] < 0) if "dias_a_vencer" in df_nopag_loc else (0.0, 0)
hoy_m, hoy_c = _agg_block(df_nopag_loc, df_nopag_loc["dias_a_vencer"] == 0) if "dias_a_vencer" in df_nopag_loc else (0.0, 0)
criticos_m = vencidos_m + hoy_m
criticos_c = vencidos_c + hoy_c
_render_cards([
    _card_html("Deuda vencida", money(vencidos_m), subtitle="Documentos atrasados", tag=_fmt_count(vencidos_c), tone="accent"),
    _card_html("Pagos que vencen hoy", money(hoy_m), subtitle="Impacto inmediato", tag=_fmt_count(hoy_c)),
    _card_html("Monto en pagos criticos", money(criticos_m), subtitle="Vencidos + hoy", tag=_fmt_count(criticos_c), tag_variant="warning"),
])


# Proyección
fig_proyeccion = None
if not df_nopag_loc.empty and "fecha_venc_30" in df_nopag_loc:
    proy = df_nopag_loc[df_nopag_loc["fecha_venc_30"].between(TODAY, TODAY + pd.to_timedelta(horizonte, "D"))]
    if not proy.empty:
        flujo = (proy.groupby(proy["fecha_venc_30"].dt.date)
                      .agg(Monto_a_Pagar=("importe_deuda", "sum"),
                           Cant_Facturas=("importe_deuda", "count"))
                      .reset_index()
                      .rename(columns={"fecha_venc_30": "Fecha"}))
        flujo["Flujo_Acumulado"] = flujo["Monto_a_Pagar"].cumsum()
        st.markdown("**Proyección Próximos 3 días**")
        small = flujo.head(3).rename(columns={"Fecha":"Día","Monto_a_Pagar":"Monto a Pagar","Cant_Facturas":"Cant. Facturas"})
        small_display = small.copy()
        small_display["Monto a Pagar"] = small_display["Monto a Pagar"].map(money)
        style_table(_table_style(small_display))
        fig = go.Figure()
        fig.add_bar(x=flujo["Fecha"], y=flujo["Monto_a_Pagar"], name="Monto Diario")
        fig.add_scatter(x=flujo["Fecha"], y=flujo["Cant_Facturas"], name="Cant. Facturas", yaxis="y2")
        fig.add_scatter(x=flujo["Fecha"], y=flujo["Flujo_Acumulado"], name="Acumulado", line=dict(dash="dash"))
        fig.update_layout(height=380, yaxis=dict(title="Monto ($)"),
                          yaxis2=dict(overlaying="y", side="right", title="Cant."))
        st.plotly_chart(fig, use_container_width=True)
        fig_proyeccion = fig
    else:
        st.info("Sin vencimientos en el horizonte seleccionado.")
else:
    st.info("No hay datos de vencimientos para proyectar con los filtros seleccionados.")

# =========================================================
# 5) Presupuesto del Día (Selección Automática)
# =========================================================
st.subheader("5) Presupuesto del Día (Selección Automática)")

base = df_nopag_loc.copy()
if base.empty or "importe_deuda" not in base:
    st.info("No hay documentos pendientes para priorizar con los filtros locales.")
else:
    base["vencida_flag"] = base["dias_a_vencer"] < 0
    base["importe_regla"] = np.where(
        base["estado_pago"].eq("autorizada_sin_pago"),
        pd.to_numeric(base.get("monto_autorizado", 0.0), errors="coerce").fillna(0.0),
        pd.to_numeric(base.get("fac_monto_total", 0.0), errors="coerce").fillna(0.0)
    )

    # --- Ordenamiento según criterio elegido ---
    base["_nivel_rank"] = base["Nivel"].map(
        {"Doc. Pendiente de Autorización": 0, "Doc. Autorizado p/ Pago": 1}
    ).fillna(2)

    # Asegurar que importe_regla sea estrictamente numérico
    base["importe_regla"] = pd.to_numeric(base["importe_regla"], errors="coerce").fillna(0.0)

    if crit_sel == "Riesgo de aprobación":
        prior = base.sort_values(
            by=["_nivel_rank", "vencida_flag", "dias_a_vencer", "importe_regla"],
            ascending=[True, False, True, False]
        )
    else:  # Urgencia de vencimiento
        prior = base.sort_values(
            by=["vencida_flag", "dias_a_vencer", "importe_regla"],
            ascending=[False, True, False]
        )

    # Monto por defecto = suma de críticos (<= 0 días)
    total_criticos = float(prior.loc[prior["dias_a_vencer"] <= 0, "importe_regla"].sum())
    default_presu  = total_criticos if total_criticos > 0 else 0.0
    monto_presu = st.number_input(
        "Monto disponible hoy",
        min_value=0.0,
        value=float(default_presu),
        step=1000.0,        # puedes subirlo si quieres saltos mayores
        format="%.0f",      # << permite ingresar 100000000 sin truncados
        key="presupuesto_hoy"
    )


    # Selección por presupuesto (corte por acumulado)
    tmp = prior.copy()
    tmp["acum"] = tmp["importe_regla"].cumsum()
    seleccion = tmp[tmp["acum"] <= float(monto_presu)].drop(columns=["acum"])

    base_keep = [
        "Nivel","prov_prioritario","cuenta_especial","fac_numero","cmp_nombre","prr_razon_social",
        "fac_fecha_factura","fecha_venc_30","dias_a_vencer","importe_regla",
        "cuenta_corriente","banco"
    ]

    def _prep_show(d: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in base_keep if c in d.columns]
        show = d[keep].rename(columns={
            "prov_prioritario":"Proveedor Prioritario","cuenta_especial":"Cuenta Especial",
            "fac_numero":"N° Factura","cmp_nombre":"Sede","prr_razon_social":"Proveedor",
            "fac_fecha_factura":"Fecha Factura","fecha_venc_30":"Fecha Venc.",
            "dias_a_vencer":"Días a Vencer","importe_regla":"Monto",
            "cuenta_corriente":"Cuenta Corriente","banco":"Banco"
        })
        if "Proveedor Prioritario" in show:
            show["Proveedor Prioritario"] = show["Proveedor Prioritario"].map({True:"Sí", False:"No"})
        if "Cuenta Especial" in show:
            show["Cuenta Especial"] = show["Cuenta Especial"].map({True:"Sí", False:"No"})
        if "Monto" in show:
            show["Monto"] = pd.to_numeric(show["Monto"], errors="coerce").fillna(0).map(money)
        return show

    def _prep_export(d: pd.DataFrame) -> pd.DataFrame:
        """Mismas columnas que la vista, pero Monto sin formato (numérico)."""
        keep = [c for c in base_keep if c in d.columns]
        out = d[keep].rename(columns={
            "prov_prioritario":"Proveedor Prioritario","cuenta_especial":"Cuenta Especial",
            "fac_numero":"N° Factura","cmp_nombre":"Sede","prr_razon_social":"Proveedor",
            "fac_fecha_factura":"Fecha Factura","fecha_venc_30":"Fecha Venc.",
            "dias_a_vencer":"Días a Vencer","importe_regla":"Monto",
            "cuenta_corriente":"Cuenta Corriente","banco":"Banco"
        })
        if "Proveedor Prioritario" in out:
            out["Proveedor Prioritario"] = out["Proveedor Prioritario"].map({True:"Sí", False:"No"})
        if "Cuenta Especial" in out:
            out["Cuenta Especial"] = out["Cuenta Especial"].map({True:"Sí", False:"No"})
        if "Monto" in out:
            out["Monto"] = pd.to_numeric(out["Monto"], errors="coerce").fillna(0.0)
        return out

    st.markdown("**Candidatas a Pago (según criterio elegido)**")
    st.dataframe(_prep_show(prior), use_container_width=True)
    st.download_button(
        "⬇️ Descargar Candidatas",
        data=excel_bytes_single(_prep_export(prior), "Candidatas"),
        file_name="candidatas_pago.xlsx", disabled=prior.empty
    )

    st.markdown(
        """
        <div class="app-note">
            <strong>Seleccion a pagar hoy</strong> -- bloque critico de pagos.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(_prep_show(seleccion), use_container_width=True)
    st.download_button(
        "⬇️ Descargar Selección de Hoy",
        data=excel_bytes_single(_prep_export(seleccion), "PagoHoy"),
        file_name="pago_hoy.xlsx", disabled=seleccion.empty
    )

    # Controles numericos
    suma_sel = float(pd.to_numeric(seleccion.get("importe_regla", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    resumen_cards = [
        _card_html("Presupuesto ingresado", money(float(monto_presu)), subtitle="Disponible hoy", tone="accent"),
        _card_html("Suma seleccion", money(suma_sel), subtitle="Total comprometido"),
        _card_html("Diferencia", money(float(monto_presu) - suma_sel), subtitle="Presupuesto - seleccion", tag_variant="warning"),
    ]
    _render_cards(resumen_cards)

# =========================================================
# 6) Reporte PDF
# =========================================================
st.subheader("6) Reporte PDF")
if st.button("📄 Generar PDF"):
    filtros = {"fac_ini": fac_ini, "fac_fin": fac_fin, "pay_ini": pay_ini, "pay_fin": pay_fin}
    pagos_criticos_pdf = pd.DataFrame()
    if 'seleccion' in locals() and isinstance(seleccion, pd.DataFrame) and not seleccion.empty:
        pagos_criticos_pdf = seleccion.rename(columns={"importe_regla":"importe_deuda"}).copy()

    pdf_bytes = generate_pdf_report(
        secciones={"kpis": True,"rankings": True,"deuda": True,"presupuesto": True},
        kpis={
            "total_facturado": money(compute_kpis(df)["total_facturado"]) if not df.empty else "-",
            "total_pagado":   money(compute_kpis(df)["total_pagado_aut"]) if not df.empty else "-",
            "dso": one_decimal(compute_kpis(df)["dso"]) if not df.empty else "-",
        },
        rankings_df=rankings_df if 'rankings_df' in locals() else pd.DataFrame(),
        niveles_kpis={},
        proyeccion_chart=fig_proyeccion if 'fig_proyeccion' in locals() else None,
        pagos_criticos_df=pagos_criticos_pdf,
        sugerencias=["Resaltar 'Selección a Pagar Hoy' como sección crítica."],
        filtros=filtros,
        presupuesto_monto=monto_presu if 'monto_presu' in locals() else None,
        seleccion_hoy_df=seleccion if 'seleccion' in locals() else None,
    )

    st.download_button(
        "Guardar PDF",
        data=pdf_bytes,
        file_name=f"Informe_Asesor_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )