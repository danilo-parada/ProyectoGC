# pages/60_Informe_Asesor.py
from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, date

from core.utils import LABELS, TOOLTIPS
from lib_common import (
    get_df_norm, general_date_filters_ui,
    advanced_filters_ui, money, one_decimal, header_ui,
    style_table, sanitize_df, safe_markdown,
)
from lib_metrics import ensure_derived_fields, compute_kpis, apply_common_filters
from lib_report import excel_bytes_single, generate_pdf_report

# -------------------- Config & Header --------------------
st.set_page_config(page_title="Informe Asesor", layout="wide")
header_ui(
    title="Informe para la Toma de Decisiones Financieras",
    current_page="Informe Asesor",
    subtitle="KPIs, deuda y priorización con foco en cuentas especiales y proveedores prioritarios"
)

# -------------------- Estilos locales para tablas --------------------
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
            ("letter-spacing", "0.6px"),
            ("padding", TABLE_ROW_PADDING),
            ("text-align", "center"),
        ]},
        {"selector": "tbody td", "props": [
            ("font-size", TABLE_FONT_SIZE),
            ("padding", TABLE_ROW_PADDING),
            ("text-align", "right"),
            ("border-bottom", "1px solid #d9e1ff"),
            ("color", "var(--app-table-body-fg)"),
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
sede, org, prov, cc, oc, est, _ = advanced_filters_ui(
    df0, show_controls=["sede", "org", "prov", "cc", "oc"]
)

estado_doc_options = [
    "Todos",
    "Autorizado sin Pago",
    "Pagado",
    "Facturado Sin Autorizar",
]
default_estado_doc = st.session_state.get("estado_doc_local", estado_doc_options[0])
if default_estado_doc not in estado_doc_options:
    default_estado_doc = estado_doc_options[0]
estado_doc_choice = st.radio(
    "Tipo de Doc./Estado (local)",
    estado_doc_options,
    horizontal=True,
    index=estado_doc_options.index(default_estado_doc),
    key="estado_doc_local",
)
estado_doc_map = {
    "Todos": None,
    "Autorizado sin Pago": "autorizada_sin_pago",
    "Pagado": "pagada",
    "Facturado Sin Autorizar": "sin_autorizacion",
}
estado_doc_value = estado_doc_map.get(estado_doc_choice)

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
df_filtered_common = df_common_no_estado
if estado_doc_value and "estado_doc" in df_filtered_common.columns:
    estado_norm = df_filtered_common["estado_doc"].astype(str)
    df_filtered_common = df_filtered_common[estado_norm == str(estado_doc_value)].copy()
else:
    df_filtered_common = df_filtered_common.copy()

df = df_common_no_estado

pagadas_mask_global = (
    df_filtered_common.get("is_pagada") if "is_pagada" in df_filtered_common.columns else None
)
if pagadas_mask_global is not None:
    df_pag = df_filtered_common[pagadas_mask_global].copy()
else:
    df_pag = df_filtered_common[df_filtered_common["estado_pago"] == "pagada"].copy()

pagadas_mask_full = df.get("is_pagada") if "is_pagada" in df.columns else None
if pagadas_mask_full is not None:
    df_nopag_all = df[~pagadas_mask_full].copy()
else:
    df_nopag_all = df[df["estado_pago"] != "pagada"].copy()

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


def _compute_presupuesto_selection(
    prior: pd.DataFrame, presupuesto: float
) -> tuple[pd.DataFrame, float, float, dict | None]:
    """Devuelve selección acumulada, suma, saldo restante y próxima factura."""

    if prior is None or prior.empty:
        return pd.DataFrame(), 0.0, float(presupuesto), None

    tmp = prior.copy()
    tmp["importe_regla_num"] = pd.to_numeric(
        tmp.get("importe_regla"), errors="coerce"
    ).fillna(0.0)
    tmp["acum"] = tmp["importe_regla_num"].cumsum()

    seleccion = tmp[tmp["acum"] <= float(presupuesto)].drop(
        columns=["acum", "importe_regla_num"], errors="ignore"
    )
    suma_sel = float(
        tmp.loc[tmp["acum"] <= float(presupuesto), "importe_regla_num"].sum()
    )

    restante_raw = float(presupuesto) - suma_sel
    restante = max(0.0, restante_raw)

    siguiente = tmp[tmp["acum"] > float(presupuesto)].head(1)
    next_info = None
    if not siguiente.empty:
        row = siguiente.iloc[0]
        next_amount = float(row.get("importe_regla_num", 0.0))
        prov_val = row.get("prr_razon_social") or row.get("Proveedor")
        prov = str(prov_val) if pd.notna(prov_val) else "Proveedor sin identificar"
        doc_val = row.get("fac_numero") or row.get("doc_numero") or row.get("doc_id")
        doc = str(doc_val) if pd.notna(doc_val) else "s/d"
        adicional = max(0.0, next_amount - restante)
        next_info = {
            "monto": next_amount,
            "proveedor": prov,
            "documento": doc,
            "adicional": adicional,
        }

    return seleccion, suma_sel, restante, next_info

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
    tooltip: str | None = None,
) -> str:
    classes = ["app-card", "app-card--frost"]
    if compact:
        classes.append("app-card__mini")
    if tone == "accent":
        classes.append("app-card--accent")
    tooltip_attr = f' title="{html.escape(str(tooltip))}"' if tooltip else ""
    title_html = html.escape(str(title))
    value_html = html.escape(str(value))
    subtitle_html = (
        f'<p class="app-card__subtitle">{html.escape(str(subtitle))}</p>'
        if subtitle
        else ""
    )
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

def _fmt_days(val: float) -> str:
    return "s/d" if pd.isna(val) else f"{one_decimal(val)} d"

def _fmt_pct(val: float) -> str:
    return "s/d" if pd.isna(val) else f"{one_decimal(val)}%"


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

# =========================================================
# 1) KPIs
# =========================================================
st.subheader("1) Resumen KPIs Facturas")
kpi_total = compute_kpis(df_filtered_common)
if df_filtered_common.empty:
    st.info("No hay datos con los filtros actuales.")
else:
    total_docs = int(kpi_total["docs_total"])
    total_pagadas = int(kpi_total["docs_pagados"])
    cards = [
        _card_html(
            "Total facturado",
            money(kpi_total["total_facturado"]),
            subtitle="Base filtrada",
            tag=f"{total_docs:,} doc.",
            tone="accent",
            tooltip=TOOLTIPS["total_facturado"],
        ),
        _card_html(
            "Desglose facturado",
            money(kpi_total["total_facturado"]),
            subtitle="Pagado vs sin pagar",
            stats=[
                ("Pagado", money(kpi_total["facturado_pagado"])),
                ("Sin pagar", money(kpi_total["facturado_sin_pagar"])),
            ],
            compact=False,
            tooltip=TOOLTIPS["desglose_facturado"],
        ),
        _card_html(
            "Total pagado (real)",
            money(kpi_total["total_pagado_real"]),
            subtitle="Facturas con pago registrado",
            tag=f"{total_pagadas:,} pagos",
            tooltip=TOOLTIPS["total_pagado_real"],
        ),
        _card_html(
            LABELS["dpp_emision_pago"],
            _fmt_days(kpi_total["dpp"]),
            subtitle=TOOLTIPS["dpp_emision_pago"],
        ),
        _card_html(
            LABELS["dic_emision_contab"],
            _fmt_days(kpi_total["dic"]),
            subtitle=TOOLTIPS["dic_emision_contab"],
        ),
        _card_html(
            LABELS["dcp_contab_pago"],
            _fmt_days(kpi_total["dcp"]),
            subtitle=TOOLTIPS["dcp_contab_pago"],
        ),
    ]
    _render_cards(cards)

    if "cuenta_especial" in df.columns:
        safe_markdown('<div class="app-separator"></div>')
        st.markdown("### Desglose por cuenta especial")
        segment_cards: list[str] = []
        for flag in (True, False):
            sub = df_filtered_common[df_filtered_common["cuenta_especial"] == flag]
            if sub.empty:
                continue
            k = compute_kpis(sub)
            stats = [
                ("Total pagado (real)", money(k["total_pagado_real"])),
                (LABELS["dpp_emision_pago"], _fmt_days(k["dpp"])),
                (LABELS["dic_emision_contab"], _fmt_days(k["dic"])),
                (LABELS["dcp_contab_pago"], _fmt_days(k["dcp"])),
            ]
            segment_cards.append(
                _card_html(
                    title=f"CE {'Si' if flag else 'No'}",
                    value=money(k["total_facturado"]),
                    subtitle=f"{int(k['docs_total']):,} doc.",
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
        "Razón Social","Monto Pagado","Días Promedio Pago",
        "Cant. Fact. ≤30 días","Cant. Fact. >30 días",
        "Proveedor Prioritario"
    ]
    if df_in.empty:
        return pd.DataFrame(columns=cols_out)

    d = df_in.copy()
    d["dias_a_pago_calc"] = pd.to_numeric(d.get("dias_a_pago_calc"), errors="coerce")
    d["monto_autorizado"] = pd.to_numeric(d.get("monto_autorizado"), errors="coerce").fillna(0.0)
    d["monto_ce"] = pd.to_numeric(d.get("monto_ce"), errors="coerce").fillna(0.0)
    d["prov_prioritario"] = d.get("prov_prioritario", False).astype(bool)
    d["cuenta_especial"] = d.get("cuenta_especial", False).astype(bool)

    grp = (
        d.groupby("prr_razon_social", dropna=False)
         .agg(
             **{
                 "Monto Pagado": ("monto_ce", "sum"),
                 "Días Promedio Pago": ("dias_a_pago_calc", lambda s: s[s >= 0].mean()),
                 "Cant. Fact. ≤30 días": ("dias_a_pago_calc", lambda s: (s <= 30).sum()),
                 "Cant. Fact. >30 días": ("dias_a_pago_calc", lambda s: (s > 30).sum()),
                 "Proveedor Prioritario": ("prov_prioritario", "mean"),
             }
         )
         .reset_index()
         .rename(columns={"prr_razon_social": "Razón Social"})
         .sort_values("Monto Pagado", ascending=False)
         .head(top_n)
    )
    grp["Proveedor Prioritario"] = grp["Proveedor Prioritario"].apply(lambda v: "Sí" if v >= 0.5 else "No")
    return grp[cols_out]

if not df_pag.empty:
    top_n = st.slider("Top N", 3, 20, 5, 1)
    rankings_df = build_top_proveedores(df_pag, top_n=top_n)
    rankings_display = rankings_df.assign(
        **{
            "Monto Pagado": rankings_df["Monto Pagado"].map(money),
            "Días Promedio Pago": rankings_df["Días Promedio Pago"].map(one_decimal),
        }
    )
    rankings_display = sanitize_df(rankings_display)
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
df_nopag_all = ensure_dias_a_vencer(ensure_importe_deuda(df_nopag_all))

def _kpis_deuda(dfin: pd.DataFrame) -> dict:
    if dfin.empty or "dias_a_vencer" not in dfin:
        return dict(vencido=0.0,c_venc=0, hoy=0.0, c_hoy=0, por_ven=0.0,c_por=0)
    vencido, c_v = _agg_block(dfin, dfin["dias_a_vencer"] < 0)
    hoy_m, c_h = _agg_block(dfin, dfin["dias_a_vencer"] == 0)
    porv, c_p = _agg_block(dfin, dfin["dias_a_vencer"] > 0)
    return dict(vencido=vencido, c_venc=c_v, hoy=hoy_m, c_hoy=c_h, por_ven=porv, c_por=c_p)

if df_nopag_all.empty:
    st.info("No hay documentos pendientes.")
else:
    n1 = df_nopag_all[df_nopag_all["Nivel"].eq("Doc. Autorizado p/ Pago")] if "Nivel" in df_nopag_all else df_nopag_all.iloc[0:0]
    n2 = df_nopag_all[df_nopag_all["Nivel"].eq("Doc. Pendiente de Autorización")] if "Nivel" in df_nopag_all else df_nopag_all.iloc[0:0]

def draw_debt_panel(title: str, dpanel: pd.DataFrame):
    safe_markdown(
        '<div class="app-title-block"><h3 style="color:#000;">'
        + html.escape(title)
        + '</h3><p>Desglose por cuenta especial</p></div>'
    )
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

_LOCAL_FILTER_STATE_KEY = "presupuesto_filters_snapshot"
_LOCAL_AMOUNT_KEY = "presupuesto_hoy"


def _update_presupuesto_session_state(filters: dict[str, str], default_value: float) -> None:
    """Sincroniza el monto disponible con los filtros locales cuando cambian."""

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

    if "monto_autorizado" in out:
        monto_aut = pd.to_numeric(out["monto_autorizado"], errors="coerce").fillna(0.0)
    else:
        monto_aut = pd.Series(0.0, index=out.index)
    if "fac_monto_total" in out:
        monto_fac = pd.to_numeric(out["fac_monto_total"], errors="coerce").fillna(0.0)
    else:
        monto_fac = pd.Series(0.0, index=out.index)

    est = out.get("estado_pago")
    if est is not None:
        autorizados = est.eq("autorizada_sin_pago")
    else:
        autorizados = pd.Series(False, index=out.index)

    out["importe_regla"] = np.where(autorizados, monto_aut, monto_fac)
    out["importe_regla"] = pd.to_numeric(out["importe_regla"], errors="coerce").fillna(0.0)

    if "Nivel" in out:
        out["_nivel_rank"] = out["Nivel"].map(
            {"Doc. Pendiente de Autorización": 0, "Doc. Autorizado p/ Pago": 1}
        ).fillna(2)
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

df_nopag_loc = _apply_local_filters(df_nopag_all)


def _apply_horizon_filter(dfin: pd.DataFrame) -> pd.DataFrame:
    if "dias_a_vencer" not in dfin:
        return dfin.copy()
    dias = pd.to_numeric(dfin["dias_a_vencer"], errors="coerce")
    return dfin[dias <= horizonte].copy()

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
        small_display = sanitize_df(small_display)
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

candidatas_base = _apply_horizon_filter(df_nopag_loc)
candidatas_prior = _prioritize_documents(candidatas_base, crit_sel)

# =========================================================
# 5) Presupuesto del Día (Selección Automática)
# =========================================================
st.subheader("5) Presupuesto del Día (Selección Automática)")

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
    for col in ("Fecha Factura", "Fecha Venc."):
        if col in show:
            fechas = pd.to_datetime(show[col], errors="coerce")
            show[col] = fechas.dt.strftime("%d-%m-%Y").fillna("s/d")
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


seleccion = pd.DataFrame()
seleccion_df = pd.DataFrame()

if candidatas_base.empty or "importe_deuda" not in candidatas_base:
    st.info("No hay documentos pendientes para priorizar con los filtros locales.")
else:
    prior = candidatas_prior

    # Monto por defecto = suma de críticos (<= 0 días)
    total_criticos = float(prior.loc[prior["dias_a_vencer"] <= 0, "importe_regla"].sum())
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
                help="Ingresa el monto en CLP. Puedes usar puntos o comas como separador de miles/decimales.",
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
        seleccion_df, suma_sel, restante, next_info = _compute_presupuesto_selection(
            prior, float(monto_presu)
        )
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
                if next_info:
                    help_text = (
                        f"Se sumarán {money(next_info['adicional'])} para incorporar la próxima factura."
                    )
                    button_label = "➕ Agregar factura extra"
                    if st.button(
                        button_label,
                        key="add_extra_invoice",
                        help=(
                            f"{next_info['proveedor']} — Factura {next_info['documento']} por {money(next_info['monto'])}. "
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
                        f"<p class='budget-panel__resume-note'>Próxima: {next_info['proveedor']} — Factura {next_info['documento']} ({money(next_info['monto'])})</p>"
                    )
                    safe_markdown(
                        f"<p class='budget-panel__resume-note budget-panel__resume-note--muted'>Ajuste necesario: {money(next_info['adicional'])}</p>"
                    )
                else:
                    safe_markdown(
                        "<p class='budget-panel__resume-note budget-panel__resume-note--muted'>No hay facturas adicionales por agregar.</p>"
                    )
        safe_markdown(
            """
                </div>
                <div class="budget-panel__helper">El valor se guarda automáticamente para esta sesión y se utilizará en el reporte PDF.</div>
            </div>
            """
        )

        # Selección por presupuesto (corte por acumulado)
        seleccion = seleccion_df

        # Controles numericos
        resumen_cards = [
            _card_html("Presupuesto ingresado", money(float(monto_presu)), subtitle="Disponible hoy", tone="accent"),
            _card_html("Suma selección", money(suma_sel), subtitle="Total comprometido"),
            _card_html("Saldo sin asignar", money(restante), subtitle="Presupuesto - selección", tag_variant="warning"),
        ]

        if next_info:
            resumen_cards.append(
                _card_html(
                    "Próxima factura a incorporar",
                    money(next_info["monto"]),
                    subtitle=f"{next_info['proveedor']} — Factura {next_info['documento']}",
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
    st.markdown("**Candidatas a Pago (todas las facturas — ordenadas por riesgo de aprobación)**")
    style_table(_table_style(candidatas_display), visible_rows=15)
    st.download_button(
        "⬇️ Descargar Candidatas",
        data=excel_bytes_single(_prep_export(candidatas_prior), "Candidatas"),
        file_name="candidatas_pago.xlsx",
        disabled=candidatas_prior.empty,
    )

with tab_seleccion:
    safe_markdown(
        """
        <div class="app-note">
            <strong>Selección a pagar hoy</strong> — bloque crítico de pagos.
        </div>
        """,
    )
    style_table(_table_style(seleccion_display), visible_rows=15)
    st.download_button(
        "⬇️ Descargar Selección de Hoy",
        data=excel_bytes_single(_prep_export(seleccion), "PagoHoy"),
        file_name="pago_hoy.xlsx",
        disabled=seleccion.empty,
    )

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
            "total_facturado": money(kpi_total["total_facturado"]) if not df_filtered_common.empty else "-",
            "total_pagado_real":   money(kpi_total["total_pagado_real"]) if not df_filtered_common.empty else "-",
            "dso": "-" if (df_filtered_common.empty or pd.isna(kpi_total["dpp"])) else one_decimal(kpi_total["dpp"]),
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
