# pages/20_Rankings.py
from __future__ import annotations
from typing import Optional, List, Union

import streamlit as st
import pandas as pd
import numpy as np
from pandas.io.formats.style import Styler

from lib_common import (
    get_df_norm, general_date_filters_ui, apply_general_filters,
    advanced_filters_ui, apply_advanced_filters, header_ui, style_table, money,
    sanitize_df, safe_markdown, collapse_sidebar_immediately,
)
from lib_metrics import ensure_derived_fields, compute_monto_pagado_real
from lib_report import excel_bytes_single, excel_bytes_multi

# ================== Estilos globales de la tabla ==================
HEADER_BG = "var(--app-primary)"   # azul corporativo global
HEADER_FG = "var(--app-table-header-fg)"   # texto encabezado global
FONT_SIZE = "var(--app-table-font-size)"
ROW_BORDER = "#d9e1ff"
ROW_STRIPED_BG = "#f2f5ff"
ROW_HOVER_BG = "#e0e8ff"

st.set_page_config(
    page_title="Rankings",
    layout="wide",
    initial_sidebar_state="collapsed",
)
collapse_sidebar_immediately()
header_ui(
    title="Rankings por Categoría",
    current_page="Rankings",
    subtitle="Top N por Proveedores y Centros, con filtros locales de Cuenta Especial y Prioritario",
    nav_active="rankings",
)

# -------- Carga base --------
df0 = get_df_norm()
if df0 is None:
    st.warning("Carga tus datos en 'Carga de Data'.")
    st.stop()

df0 = ensure_derived_fields(df0)
df0["monto_pagado_real"] = compute_monto_pagado_real(df0)

# -------- Filtros globales (sin prioritario global) --------
fac_ini, fac_fin, pay_ini, pay_fin = general_date_filters_ui(df0)
sede, org, prov, cc, oc, est, _prio_removed = advanced_filters_ui(
    df0, show_controls=['sede','org','prov','cc','oc']
)
df = apply_general_filters(df0, fac_ini, fac_fin, pay_ini, pay_fin)
df = apply_advanced_filters(df, sede, org, prov, cc, oc, est, prio=[])

# Trabajamos con pagadas segun monto pagado real
pagadas_series = pd.to_numeric(df.get("monto_pagado_real"), errors="coerce") if "monto_pagado_real" in df else pd.Series(0.0, index=df.index)
pagadas_series = pagadas_series.fillna(0.0)
dfp = df[pagadas_series > 0].copy()
if dfp.empty:
    st.info("No hay facturas pagadas con los filtros actuales.")
    st.stop()

# -------- Filtros LOCALES (solo esta página) --------
col_ce, col_prio, col_topn, col_order = st.columns([1, 1, 1, 1.3], gap="small")

# Cuenta Especial (local)
ce_local = "Todas"
if "cuenta_especial" in dfp.columns:
    ce_local = col_ce.radio(
        "Cuenta Especial (local)",
        options=["Todas", "Cuenta Especial", "No Cuenta Especial"],
        horizontal=True, index=0
    )

# Prioritario (local)
prio_local = "Todos"
if "prov_prioritario" in dfp.columns:
    prio_local = col_prio.radio(
        "Proveedor Prioritario (local)",
        options=["Todos", "Prioritario", "No Prioritario"],
        horizontal=True, index=0
    )

def _apply_local_filters(dfin: pd.DataFrame) -> pd.DataFrame:
    out = dfin
    if "cuenta_especial" in out.columns:
        if ce_local == "Cuenta Especial":
            out = out[out["cuenta_especial"] == True]
        elif ce_local == "No Cuenta Especial":
            out = out[out["cuenta_especial"] == False]
    if "prov_prioritario" in out.columns:
        if prio_local == "Prioritario":
            out = out[out["prov_prioritario"] == True]
        elif prio_local == "No Prioritario":
            out = out[out["prov_prioritario"] == False]
    return out

dfp_f = _apply_local_filters(dfp)

# -------- Parámetros de Ranking --------
top_n = col_topn.slider("Seleccionar Top N", 5, 50, 20, 1)
orden_metric = col_order.radio(
    "Ordenar por:",
    ["Monto Pagado", "Cantidad Documentos"],
    horizontal=True, index=0
)

# -------- Agregaciones --------
def _agg_base(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Devuelve agregación por grupo con:
      - Monto Pagado (real) / Cantidad Documentos
      - Cantidad Prioritario / % Prioritario (1 decimal)
      - Cantidad Cuenta Especial / % Cuenta Especial (1 decimal)
      - Mayoría: Proveedor Prioritario (Sí/No), Cuenta Especial (Sí/No)
    """
    if df_in.empty:
        return pd.DataFrame(columns=[
            group_col, "Monto Pagado", "Cantidad Documentos",
            "Cantidad Prioritario","% Prioritario",
            "Cantidad Cuenta Especial","% Cuenta Especial",
            "Proveedor Prioritario","Cuenta Especial"
        ])

    g = (df_in
         .assign(
             prov_prioritario=df_in.get("prov_prioritario", False).astype(bool),
             cuenta_especial=df_in.get("cuenta_especial", False).astype(bool),
             monto_pagado_real=pd.to_numeric(df_in.get("monto_pagado_real", 0.0), errors="coerce").fillna(0.0),
         )
         .groupby(group_col, dropna=False)
         .agg(
             **{
                 "Monto Pagado": ("monto_pagado_real", "sum"),
                 "Cantidad Documentos": (group_col, "count"),
                 "Cantidad Prioritario": ("prov_prioritario", "sum"),
                 "Cantidad Cuenta Especial": ("cuenta_especial", "sum"),
             }
         )
         .reset_index()
    )

    # % internos al grupo (1 decimal)
    g["% Prioritario"] = np.where(
        g["Cantidad Documentos"] > 0,
        (g["Cantidad Prioritario"] / g["Cantidad Documentos"]) * 100.0, 0.0
    ).round(1)

    g["% Cuenta Especial"] = np.where(
        g["Cantidad Documentos"] > 0,
        (g["Cantidad Cuenta Especial"] / g["Cantidad Documentos"]) * 100.0, 0.0
    ).round(1)

    # Mayoría (≥ 50%)
    g["Proveedor Prioritario"] = g["% Prioritario"].apply(lambda p: "Sí" if p >= 50.0 else "No")
    g["Cuenta Especial"] = g["% Cuenta Especial"].apply(lambda p: "Sí" if p >= 50.0 else "No")

    # Orden final
    g = g.sort_values(orden_metric, ascending=False).reset_index(drop=True)
    return g

def agregar_ranking(
    df_in: pd.DataFrame,
    group_col: str,
    nombre_col: str,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    agg = _agg_base(df_in, group_col)
    agg = agg.rename(columns={group_col: nombre_col})
    agg = agg.head(top_n)
    # Orden amigable de columnas
    cols = [
        nombre_col,
        "Monto Pagado","Cantidad Documentos",
        "Cantidad Prioritario","% Prioritario",
        "Cantidad Cuenta Especial","% Cuenta Especial",
        "Proveedor Prioritario","Cuenta Especial"
    ]
    if drop_cols:
        cols = [c for c in cols if c not in drop_cols]
    agg = agg.reindex(columns=cols)
    return agg

def _format_percent_cols_for_display(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df_disp = df_in.copy()
    for c in cols:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else v)
    return df_disp

def _format_money_cols_for_display(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Aplica formato contable local (usa lib_common.money):
      - Signo $,
      - Miles con punto,
      - Decimales con coma (si aplica).
    Solo para mostrar en Streamlit.
    """
    df_disp = df_in.copy()
    for c in cols:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].apply(lambda v: money(v) if pd.notnull(v) else v)
    return df_disp

def _style_headers(df_disp: Union[pd.DataFrame, Styler]):
    """
    Resalta encabezados (azul rey, texto blanco), agranda fuente (FONT_SIZE),
    y alinea valores a la derecha (estilo contable). No afecta exportación.
    """
    if isinstance(df_disp, pd.DataFrame):
        sty = df_disp.style
    else:
        sty = df_disp

    sty = sty.hide(axis="index")
    sty = sty.set_table_styles([
        {"selector": "thead tr", "props": [
            ("background-color", HEADER_BG),
            ("color", HEADER_FG),
            ("font-weight", "bold"),
            ("font-size", FONT_SIZE),
            ("text-align", "center"),
            ("border-radius", "12px 12px 0 0")
        ]},
        {"selector": "th", "props": [
            ("background-color", "transparent"),
            ("color", HEADER_FG),
            ("font-weight", "600"),
            ("font-size", FONT_SIZE),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.5px"),
            ("padding", "14px 18px"),
            ("text-align", "center")
        ]},
        {"selector": "tbody td", "props": [
            ("font-size", FONT_SIZE),
            ("padding", "14px 18px"),
            ("text-align", "right"),
            ("border-bottom", f"1px solid {ROW_BORDER}"),
            ("color", "var(--app-table-body-fg)")
        ]},
        {"selector": "tbody tr:nth-child(even)", "props": [
            ("background-color", ROW_STRIPED_BG)
        ]},
        {"selector": "tbody tr:hover", "props": [
            ("background-color", ROW_HOVER_BG)
        ]},
        {"selector": "tbody td:first-child", "props": [
            ("text-align", "left"),
            ("font-weight", "600"),
            ("color", "#1f2a55")
        ]}
    ], overwrite=False)

    return sty

# -------- Separador --------
safe_markdown("---")

# -------- Top Proveedores --------
safe_markdown("---")
st.subheader("Top Proveedores")
prov = agregar_ranking(
    dfp_f,
    "prr_razon_social",
    "Razón Social",
    drop_cols=["Cantidad Prioritario", "% Prioritario", "Cuenta Especial"],
)
# Formato de pantalla (no afecta Excel)
prov_disp = _format_percent_cols_for_display(prov, ["% Cuenta Especial"])
prov_disp = _format_money_cols_for_display(prov_disp, ["Monto Pagado"])
prov_disp = sanitize_df(prov_disp)
style_table(_style_headers(prov_disp))
st.download_button(
    "⬇️ Descargar Ranking de Proveedores",
    data=excel_bytes_single(prov, "RankingProveedores"),
    file_name="ranking_proveedores.xlsx"
)

# -------- Top Centros de Costo --------
safe_markdown("---")
st.subheader("Top Centros de Costo")
if "nombre_centro_costo" in dfp_f.columns:
    cc = agregar_ranking(
        dfp_f,
        "nombre_centro_costo",
        "Centro de Costo",
        drop_cols=["Cantidad Cuenta Especial", "% Cuenta Especial", "Proveedor Prioritario"],
    )
    cc_disp = _format_percent_cols_for_display(cc, ["% Prioritario"])
    cc_disp = _format_money_cols_for_display(cc_disp, ["Monto Pagado"])
    cc_disp = sanitize_df(cc_disp)
    style_table(_style_headers(cc_disp))
    st.download_button(
        "⬇️ Descargar Ranking de Centros de Costo",
        data=excel_bytes_single(cc, "RankingCC"),
        file_name="ranking_cc.xlsx"
    )
else:
    cc = None
    st.info("No hay datos de 'Centro de Costo' en la base actual.")

# -------- Resumen Global (para Excel) --------
def _resumen_dim(df_in: pd.DataFrame, flag_col: str, nombre_si: str, nombre_no: str) -> pd.DataFrame:
    if flag_col not in df_in.columns:
        return pd.DataFrame(columns=[
            "Categoría", "Cantidad Documentos", "Monto Contabilizado", "Monto Pagado",
            "% Docs Total","% Monto Contab. Total","% Monto Pag. Total"
        ])

    tmp = df_in.assign(
        monto_autorizado=pd.to_numeric(df_in.get("monto_autorizado", 0.0), errors="coerce").fillna(0.0),
        monto_pagado_real=pd.to_numeric(df_in.get("monto_pagado_real", 0.0), errors="coerce").fillna(0.0),
    )

    total_docs = len(tmp)
    total_aut = float(tmp["monto_autorizado"].sum())
    total_pag = float(tmp["monto_pagado_real"].sum())

    g = (tmp.groupby(flag_col)
             .agg(
                 **{
                     "Cantidad Documentos": (flag_col, "count"),
                     "Monto Contabilizado": ("monto_autorizado", "sum"),
                     "Monto Pagado": ("monto_pagado_real", "sum"),
                 }
             )
             .reset_index()
             .replace({True: nombre_si, False: nombre_no})
             .rename(columns={flag_col: "Categoría"})
    )

    # % sobre el total (1 decimal)
    g["% Docs Total"] = np.where(total_docs>0, (g["Cantidad Documentos"]/total_docs)*100.0, 0.0).round(1)
    g["% Monto Contab. Total"] = np.where(total_aut>0, (g["Monto Contabilizado"]/total_aut)*100.0, 0.0).round(1)
    g["% Monto Pag. Total"] = np.where(total_pag>0, (g["Monto Pagado"]/total_pag)*100.0, 0.0).round(1)

    # Orden amigable
    cols = [
        "Categoría","Cantidad Documentos","Monto Contabilizado","Monto Pagado",
        "% Docs Total","% Monto Contab. Total","% Monto Pag. Total"
    ]
    g = g.reindex(columns=cols)
    return g

def resumen_global(dfin: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    out["Resumen_Prioritario"] = _resumen_dim(dfin, "prov_prioritario", "Prioritario", "No Prioritario")
    out["Resumen_CuentaEspecial"] = _resumen_dim(dfin, "cuenta_especial", "Cuenta Especial", "No Cuenta Especial")
    return out

# -------- Exportar todo a Excel (multi-hoja) --------
safe_markdown("---")
st.subheader("Exportar Resultados de Rankings a Excel")

sheets = {"RankingProveedores": prov}
if cc is not None:
    sheets["RankingCC"] = cc
sheets.update(resumen_global(dfp_f))

st.download_button(
    "⬇️ Descargar Excel Completo de Rankings",
    data=excel_bytes_multi(sheets),
    file_name="rankings_completo.xlsx"
)