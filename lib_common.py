# lib_common.py
from __future__ import annotations
import html
import re
import warnings
warnings.filterwarnings("ignore")

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st

_DIV_CLOSE_RE = re.compile(r'^\s*</div>\s*$', re.I)
_DIV_TAG_RE = re.compile(r'^\s*</?div>\s*$', re.I)


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza celdas que contienen exclusivamente etiquetas ``<div>``."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    return df.replace({_DIV_TAG_RE: ""}, regex=True)


def safe_markdown(html: str):
    """Evita imprimir cierres huérfanos y HTML desbalanceado en bloques cortos."""

    if _DIV_CLOSE_RE.match(html or ""):
        return
    normalized = (html or "")
    opens = normalized.lower().count("<div")
    closes = normalized.lower().count("</div>")
    if closes > opens:
        return
    st.markdown(normalized, unsafe_allow_html=True)

# ============================================================
# 1) Columnas esperadas en la BBDD principal (facturas)
# ============================================================
EXPECTED_COLS = [
    "origen","fac_correlativo","fac_numero","pdp_ano_pptario","cmp_codigo",
    "cmp_nombre","fac_fecha_factura","fac_monto_total","prr_razon_social",
    "estado","prr_rut_real","prr_dv_razon_social","monto_autorizado",
    "fecha_autoriza","cc","fecha_cc","monto_pagado","fecha_pagado","ce",
    "dias_corridos","fecha_gasto","codigo_proveedor","factoring",
    "codigo_centro_costo","nombre_centro_costo","con_oc","ap",
]

HONORARIOS_EXPECTED_MAP = {
    "cnv_correlativo_actualizacion": "fac_correlativo",
    "numero_documento": "fac_numero",
    "fecha_emision": "fac_fecha_factura",
    "fecha_cuota": "fecha_cc",
    "fecha_ce": "fecha_pagado",
    "dcu_monto": "fac_monto_total",
    "monto_cuota": "monto_autorizado",
    "liquido_cuota": "monto_pagado",
    "centro_costo_costeo": "codigo_centro_costo",
    "nombre_centro": "nombre_centro_costo",
    "estado_cuota": "estado",
    "cnv_fecha_inicio": "fecha_autoriza",
    "cnv_fecha_termino": "fecha_gasto",
}
HONORARIOS_DATE_COLS = [
    "fecha_ce",
    "fecha_emision",
    "fecha_cuota",
    "cnv_fecha_inicio",
    "cnv_fecha_termino",
]

HONORARIOS_VALID_ESTADOS = {
    "REGISTRADA",
    "BOLETA HONORARIO REGISTRADA",
    "BOLETA HONORARIO ENVIADA",
    "INFORME ENVIADO",
    "INFORME AUTORIZADO",
    "AUTORIZADA",
    "BOLETA HONORARIO AUTORIZADA",
    "AUTORIZADA A PAGO",
    "CONTABILIZADA",
    "PAGADA",
}

HONORARIOS_ESTADO_LOOKUP = {val.lower(): val for val in HONORARIOS_VALID_ESTADOS}

# ============================================================
# 2) Etiquetas legibles para Estado / Tipo de Documento
# ============================================================
ESTADO_LABEL: Dict[str, str] = {
    "pagada": "Pagado",
    "autorizada_sin_pago": "Autorizado sin Pago",
    "sin_autorizacion": "Facturado Sin Autorizar",
}
_INV_ESTADO_LABEL: Dict[str, str] = {v: k for k, v in ESTADO_LABEL.items()}

# ============================================================
# 3) Utilidades de formato / UI
# ============================================================
def money(val: float) -> str:
    """Formatea un número en estilo contable con punto para miles y coma para decimales."""
    try:
        # Formatea con separador inglés (1,234,567.89) y luego intercambia
        return f"${val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(val)

def one_decimal(val: float) -> str:
    """Formatea un número con 1 decimal, estilo latino (punto miles, coma decimales)."""
    try:
        return f"{val:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(val)

def read_any(file)->pd.DataFrame:
    """
    Lee CSV o Excel desde streamlit.file_uploader o ruta.
    - CSV: autodetecta separador con engine='python'
    - Excel: primera hoja
    """
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, sep=None, engine="python", encoding="utf-8", low_memory=False)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        raise ValueError(f"Formato de archivo no soportado: {name}")

def style_table(
    df: pd.DataFrame | Styler,
    use_container_width: bool = True,
    height: int = 420,
    *,
    visible_rows: int | None = None,
):
    """Renderiza una tabla con estilo consistente.

    Cuando se recibe un ``Styler`` se utiliza la versión HTML con estilos
    personalizados. En ese caso ``visible_rows`` permite fijar una altura
    máxima equivalente al número de filas visibles antes de que aparezca una
    barra de desplazamiento vertical.
    """
    header_height = 64
    row_height = 52

    if isinstance(df, Styler):
        try:
            df = df.copy()
        except AttributeError:
            pass
        try:
            df.data = sanitize_df(df.data)  # type: ignore[attr-defined]
        except AttributeError:
            pass
        safe_markdown(
            """
            <style>
            .styled-table-wrapper {
                width: 100%;
                overflow-x: auto;
                border-radius: 16px;
                box-shadow: 0 12px 30px rgba(15, 34, 75, 0.12);
                border: 1px solid #d9e1ff;
                margin-bottom: 1.5rem;
                background-color: #ffffff;
            }
            .styled-table-wrapper table {
                width: 100% !important;
                border-collapse: separate !important;
                border-spacing: 0;
                color: #000000;
            }
            .styled-table-wrapper td {
                color: #000000 !important;
                opacity: 1 !important;
                font-weight: 500;
            }
            .styled-table-wrapper th {
                color: #000000 !important;
                font-weight: 700;
            }
            </style>
            """
        )
        extra_style = ""
        if visible_rows is not None and visible_rows > 0:
            max_height = header_height + visible_rows * row_height
            extra_style = f"max-height:{max_height}px; overflow-y:auto;"
        table_html = df.to_html()
        # Algunas tablas generadas por pandas muestran un literal "<div></div>"
        # como texto en celdas vacías. Eliminamos esos fragmentos o, en su
        # defecto, los reemplazamos por un contenedor vacío sin contenido
        # visible para que no aparezcan en la interfaz.
        for unwanted in ("&lt;div&gt;&lt;/div&gt;", "<div></div>"):
            if unwanted in table_html:
                table_html = table_html.replace(unwanted, "")
        safe_markdown(
            f"""
            <div class="styled-table-wrapper" style="{extra_style}">
                {table_html}
            </div>
            """
        )
    else:
        if visible_rows is not None and visible_rows > 0:
            height = header_height + visible_rows * row_height
        df = sanitize_df(df)
        st.dataframe(df, use_container_width=use_container_width, height=height)

# ============================================================
# 4) Tema visual y cabecera
# ============================================================
_THEME_CSS_PATH = Path(__file__).resolve().parent / "styles" / "theme.css"
_THEME_CSS_CACHE: str | None = None

def load_ui_theme():
    """
    Injecta la hoja de estilos global en cada render para mantener la apariencia
    moderna incluso al navegar entre paginas.
    """
    global _THEME_CSS_CACHE
    if _THEME_CSS_CACHE is None:
        try:
            _THEME_CSS_CACHE = _THEME_CSS_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            _THEME_CSS_CACHE = ""
    if _THEME_CSS_CACHE:
        safe_markdown(f"<style>{_THEME_CSS_CACHE}</style>")

def header_ui(title: str, current_page: str, subtitle: str | None = None):
    load_ui_theme()
    safe_title = html.escape(str(title))
    safe_page = html.escape(str(current_page))
    sub_html = f'<p class="app-hero__subtitle">{html.escape(str(subtitle))}</p>' if subtitle else ""
    safe_markdown(
        f"""
        <div class="app-hero">
            <div class="app-hero__titles">
                <h1>{safe_title}</h1>
                {sub_html}
            </div>
            <div class="app-hero__badge">{safe_page}</div>
        </div>
        """
    )


# ============================================================
# 5) Estado de sesion y setup basico
# ============================================================
def init_session_keys():
    """
    Inicializa claves de sesión esperadas por la app.
    (No pisa valores existentes)
    """
    ss = st.session_state
    defaults = {
        "df_raw": None,          # última carga cruda (opcional)
        "df": None,              # base normalizada vigente (deduplicada)
        "honorarios": None,        # base de honorarios normalizada
        "df_honorarios_raw": None, # honorarios original
        "honorarios_enriquecido": None,
        "map_ok": False, "col_map": {},
        "fac_ini": None, "fac_fin": None, "pay_ini": None, "pay_fin": None,
        "sede_sel": [], "org_sel": [], "prov_sel": [], "cc_sel": [], "oc_sel": [],
        "est_sel": [], "prio_sel": [],
        # Maestras:
        "df_prio_raw": None,         # proveedores prioritarios (original)
        "df_ctaes_raw": None,        # cuentas especiales (original)
        "prio_keys": set(),          # claves normalizadas (codigo_proveedor)
        "ctaes_keys": set(),         # claves normalizadas (codigo_contable)
        # Resumenes:
        "_match_summary": None,      # ultimo resumen de match (dict)
        "_match_timestamp": None,    # timestamp de ultimo calculo de facturas
        "_match_timestamp_view": None, # timestamp mostrado en UI
        "_facturas_count": 0,        # contador de facturas normalizadas
        "df_cache": None,            # copia del ultimo dataframe normalizado
        "_honorarios_cargados": False, # bandera de honorarios cargados
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

def get_df_norm()->pd.DataFrame|None:
    """Devuelve la base normalizada vigente en sesión (deduplicada)."""
    return st.session_state.get("df")

# ============================================================
# 5) Normalización de claves para match robusto
# ============================================================
def _norm_key_series(s: pd.Series) -> pd.Series:
    """Normaliza claves a texto: strip, quita espacios internos y minúscula."""
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", "", regex=True)
         .str.lower()
    )

# ============================================================
# 6) Registro de tablas auxiliares (maestras)
# ============================================================
def load_proveedores_prioritarios(df_prov: pd.DataFrame, col_codigo: str = "codigo_proveedor"):
    """Registra en sesión los proveedores prioritarios."""
    ss = st.session_state
    if df_prov is None or col_codigo not in df_prov:
        ss["prio_keys"] = set(); ss["df_prio_raw"] = None; return
    dfp = df_prov.copy()
    dfp["key_proveedor"] = _norm_key_series(dfp[col_codigo])
    ss["prio_keys"] = set(dfp["key_proveedor"].dropna().unique().tolist())
    ss["df_prio_raw"] = dfp

def load_cuentas_especiales(df_ctas: pd.DataFrame, col_codigo_contable: str = "codigo_contable"):
    """Registra en sesión la maestra de Cuentas Especiales (bancos)."""
    ss = st.session_state
    if df_ctas is None or col_codigo_contable not in df_ctas:
        ss["ctaes_keys"] = set(); ss["df_ctaes_raw"] = None; return
    dft = df_ctas.copy()
    dft["key_cc"] = _norm_key_series(dft[col_codigo_contable])
    ss["ctaes_keys"] = set(dft["key_cc"].dropna().unique().tolist())
    ss["df_ctaes_raw"] = dft

# ============================================================
# 7) Honorarios
# ============================================================
def _clean_estado_cuota(series, index) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="object")

    def _canon(value):
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        if not text:
            return pd.NA
        return HONORARIOS_ESTADO_LOOKUP.get(text.lower(), pd.NA)

    cleaned = pd.Series(series, index=index, copy=True).apply(_canon)
    return cleaned.astype("string")


def clean_estado_cuota(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza la columna estado_cuota eliminando valores vacios."""
    if df is None or getattr(df, "empty", True):
        return df
    col = "estado_cuota" if "estado_cuota" in df.columns else None
    if col is None and "estado" in df.columns:
        col = "estado"
    if col is None:
        return df
    cleaned = _clean_estado_cuota(df[col], df.index)
    mask = cleaned.notna()
    df_out = df.loc[mask].copy()
    df_out["estado_cuota"] = cleaned.loc[mask].astype("string")
    return df_out


def merge_honorarios_con_bancos(df_honorarios: pd.DataFrame, df_bancos: pd.DataFrame) -> pd.DataFrame:
    """Aplica un left join entre honorarios y bancos especiales."""
    if df_honorarios is None or getattr(df_honorarios, "empty", True):
        return df_honorarios
    if df_bancos is None or getattr(df_bancos, "empty", True):
        return df_honorarios.copy()
    left_key = next((col for col in ("centro_costo_costeo", "codigo_centro_costo") if col in df_honorarios.columns), None)
    right_key = next((col for col in ("codigo_contable", "key_cc") if col in df_bancos.columns), None)
    if left_key is None or right_key is None:
        return df_honorarios.copy()
    left = df_honorarios.copy()
    right = df_bancos.copy()
    left["_merge_key_cc"] = _norm_key_series(left[left_key])
    right["_merge_key_cc"] = _norm_key_series(right[right_key])
    cols_bancos = ["_merge_key_cc"]
    for col in ["codigo_contable", "ano_proyecto", "proyecto", "sede", "sede_pago", "banco", "cuenta_corriente", "cuenta_contable", "cuenta_cont_descripcion"]:
        if col in right.columns and col not in cols_bancos:
            cols_bancos.append(col)
    right_subset = right[cols_bancos].drop_duplicates("_merge_key_cc")
    merged = left.merge(right_subset, on="_merge_key_cc", how="left", suffixes=("", "_cta"))
    return merged.drop(columns="_merge_key_cc")


def load_honorarios(df_hon: pd.DataFrame) -> pd.DataFrame | None:
    """Normaliza y registra la base de honorarios en sesion."""
    ss = st.session_state
    if df_hon is None or df_hon.empty:
        ss["honorarios"] = None
        ss["df_honorarios_raw"] = None
        ss["honorarios_enriquecido"] = None
        return None

    df = df_hon.copy()

    for col in HONORARIOS_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for src, target in HONORARIOS_EXPECTED_MAP.items():
        if src in df.columns:
            df[target] = df[src]

    if "liquido" in df.columns and "monto_pagado" not in df.columns:
        df["monto_pagado"] = df["liquido"]

    df["origen"] = df.get("origen", "HONORARIOS")

    df = clean_estado_cuota(df)

    if "codigo_centro_costo" in df.columns:
        df["codigo_centro_costo"] = df["codigo_centro_costo"].astype(str)
        df["codigo_centro_costo"] = df["codigo_centro_costo"].replace({"nan": np.nan, "None": np.nan})

    if "nombre_centro" in df.columns and "nombre_centro_costo" not in df.columns:
        df["nombre_centro_costo"] = df["nombre_centro"]

    df_norm = normalize_types(df)

    if "fecha_ce" in df.columns:
        df_norm["fecha_ce"] = pd.to_datetime(df["fecha_ce"], errors="coerce")
    else:
        df_norm["fecha_ce"] = pd.to_datetime(df_norm.get("fecha_pagado"), errors="coerce")

    if "fecha_cuota" in df.columns:
        df_norm["fecha_cuota"] = pd.to_datetime(df["fecha_cuota"], errors="coerce")

    if "nombre_centro" not in df_norm.columns and "nombre_centro_costo" in df_norm.columns:
        df_norm["nombre_centro"] = df_norm["nombre_centro_costo"]

    df_norm = clean_estado_cuota(df_norm)

    if "codigo_centro_costo" in df_norm.columns:
        df_norm["codigo_centro_costo"] = df_norm["codigo_centro_costo"].astype(str)
        df_norm["codigo_centro_costo"] = df_norm["codigo_centro_costo"].replace({"nan": np.nan, "None": np.nan})

    ss["df_honorarios_raw"] = df_hon
    ss["honorarios"] = df_norm
    ss["honorarios_enriquecido"] = None
    ss["_honorarios_cargados"] = True
    return df_norm


# Helpers honorarios
def get_honorarios_df() -> pd.DataFrame | None:
    df = st.session_state.get("honorarios")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return df if isinstance(df, pd.DataFrame) else None


def reset_honorarios():
    for k in ["honorarios", "df_honorarios_raw", "honorarios_enriquecido", "honorarios_summary"]:
        if k in st.session_state:
            st.session_state[k] = None
    st.session_state["_honorarios_cargados"] = False

def reset_proveedores():
    """Elimina proveedores prioritarios cargados en memoria."""
    ss = st.session_state
    if "df_prio_raw" in ss:
        ss["df_prio_raw"] = None
    if "prio_keys" in ss:
        ss["prio_keys"] = set()
    if "_match_summary" in ss:
        ss["_match_summary"] = None
    if "facturas_summary" in ss:
        ss["facturas_summary"] = None

def reset_cuentas_especiales():
    """Elimina la maestra de cuentas especiales en memoria."""
    ss = st.session_state
    if "df_ctaes_raw" in ss:
        ss["df_ctaes_raw"] = None
    if "ctaes_keys" in ss:
        ss["ctaes_keys"] = set()
    if "_match_summary" in ss:
        ss["_match_summary"] = None
    if "facturas_summary" in ss:
        ss["facturas_summary"] = None
    if "honorarios_summary" in ss:
        ss["honorarios_summary"] = None
# ============================================================
# 8) Mapeo de columnas y normalización base
# ============================================================
def mapping_ui(df_cols)->tuple[dict,bool]:
    with st.expander("▼ 1. Mapeo de Columnas (Obligatorio)", expanded=False):
        col_map = {}
        df_cols = list(map(str, df_cols))
        col1, col2 = st.columns(2)
        for i, exp in enumerate(EXPECTED_COLS):
            target = col1 if i < len(EXPECTED_COLS)/2 else col2
            opciones = ["— (no usar)"] + df_cols
            idx = opciones.index(exp) if exp in df_cols else 0
            sel = target.selectbox(f"Asignar → **{exp}**", opciones, index=idx, key=f"map_{exp}")
            col_map[exp] = None if sel == "— (no usar)" else sel
    ok = col_map.get("fac_fecha_factura") is not None
    st.caption("Estado del Mapeo: " + ("✅ Correcto" if ok else "❗ Falta 'fac_fecha_factura'"))
    return col_map, ok

def apply_mapping(df_raw: pd.DataFrame, col_map: dict)->pd.DataFrame:
    rename = {v:k for k,v in col_map.items() if v}
    df = df_raw.rename(columns=rename).copy()
    return normalize_types(df)

def _to_date(s):
    for fmt in ["%d/%m/%Y","%Y-%m-%d","%d-%m-%Y"]:
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

def normalize_types(df: pd.DataFrame)->pd.DataFrame:
    """
    Normaliza tipos + deriva estado + marca PRIORITARIO y CUENTA ESPECIAL,
    y arrastra columnas de la maestra de cuentas especiales.
    """
    ss = st.session_state
    df = df.copy()

    # Fechas
    for c in ["fac_fecha_factura","fecha_autoriza","fecha_pagado","fecha_gasto","fecha_cc"]:
        if c in df.columns: df[c]=_to_date(df[c])

    # Numéricos
    for c in ["fac_monto_total","monto_autorizado","monto_pagado","con_oc",
              "fac_correlativo","pdp_ano_pptario","prr_rut_real","dias_corridos"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c], errors="coerce")

    # Claves normalizadas
    if "codigo_proveedor" in df.columns:
        df["key_proveedor"] = _norm_key_series(df["codigo_proveedor"])
    if "codigo_centro_costo" in df.columns:
        df["key_cc"] = _norm_key_series(df["codigo_centro_costo"])

    # Estado de documento
    mpag = df.get("monto_pagado", pd.Series(0, index=df.index)).fillna(0)
    maut = df.get("monto_autorizado", pd.Series(0, index=df.index)).fillna(0)
    df["estado_pago"] = np.select(
        [mpag>0, (maut>0)&(mpag==0), maut==0],
        ["pagada","autorizada_sin_pago","sin_autorizacion"],
        default="sin_autorizacion"
    )

    # Derivados de tiempo
    if "fac_fecha_factura" in df.columns and "fecha_autoriza" in df.columns:
        df["dias_factura_autorizacion"] = (df["fecha_autoriza"]-df["fac_fecha_factura"]).dt.days
    if "fac_fecha_factura" in df.columns and "fecha_pagado" in df.columns:
        df["dias_a_pago_calc"] = (df["fecha_pagado"]-df["fac_fecha_factura"]).dt.days
    else:
        df["dias_a_pago_calc"] = np.nan
    if "fecha_autoriza" in df.columns and "fecha_pagado" in df.columns:
        df["dias_autorizacion_pago_calc"] = (df["fecha_pagado"]-df["fecha_autoriza"]).dt.days

    hoy = pd.Timestamp(date.today())
    df["dias_transcurridos_estado"] = np.where(
        df["estado_pago"].eq("pagada"),
        df["dias_a_pago_calc"],
        (hoy-df["fac_fecha_factura"]).dt.days if "fac_fecha_factura" in df.columns else np.nan
    )

    # Flags desde maestras (sets)
    prio_set = set(ss.get("prio_keys", set()))
    ctaes_set = set(ss.get("ctaes_keys", set()))
    df["prov_prioritario"] = False
    df["cuenta_especial"] = False
    if "key_proveedor" in df.columns and prio_set:
        df.loc[df["key_proveedor"].isin(prio_set), "prov_prioritario"] = True
    if "key_cc" in df.columns and ctaes_set:
        df.loc[df["key_cc"].isin(ctaes_set), "cuenta_especial"] = True

    # ---------- MERGE con maestra de cuentas (sin multiplicar filas) ----------
    len_before = len(df)
    df_ctaes = ss.get("df_ctaes_raw")
    extra_cols = ["cuenta_contable","cuenta_corriente","banco","sede_pago",
                  "cuenta_cont_descripcion","proyecto","ano_proyecto"]
    for c in extra_cols:
        if c not in df.columns: df[c] = np.nan

    if isinstance(df_ctaes, pd.DataFrame) and not df_ctaes.empty:
        dft = df_ctaes.copy()
        # clave normalizada
        if "key_cc" not in dft.columns:
            if "codigo_contable" in dft.columns:
                dft["key_cc"] = _norm_key_series(dft["codigo_contable"])
            else:
                dft["key_cc"] = ""
        # 1) QUEDARSE CON UNA SOLA FILA POR key_cc
        cols_for_order = ["key_cc"] + [c for c in extra_cols if c in dft.columns]
        dft = dft[cols_for_order].copy()
        # si hay múltiples filas para el mismo key_cc, nos quedamos con la primera
        dft = dft.drop_duplicates(subset=["key_cc"], keep="first")

        # 2) MERGE 1→1
        df = df.merge(dft, on="key_cc", how="left", suffixes=("", "_m"))
        # completar columnas extra (si quedaron en _m)
        for c in extra_cols:
            cm = c + "_m"
            if cm in df.columns:
                df[c] = df[c].fillna(df[cm])

        # 3) Limpieza de columnas temporales
        drop_aux = [c for c in df.columns if c.endswith("_m")]
        if drop_aux:
            df = df.drop(columns=drop_aux)

        # 4) SALVAGUARDA: si por algún motivo se multiplicó, volvemos al tamaño original
        if len(df) > len_before:
            base_cols = [c for c in df.columns if c not in extra_cols]
            df = df.drop_duplicates(subset=base_cols, keep="first")

    return df

# ============================================================
# 8) Deduplicación + Registro de documentos
# ============================================================
def deduplicate_docs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina únicamente filas idénticas (duplicados exactos en todas las columnas).
    Esto evita borrar documentos válidos que comparten correlativo/clave.
    """
    if df is None or df.empty:
        return df
    mask = ~df.astype(str).duplicated(keep="first")
    return df.loc[mask].copy()

def register_documents(df: pd.DataFrame, dedup: bool = False):
    """
    Registra la base normalizada como 'df' vigente en sesión.
    Si dedup=True, elimina sólo duplicados exactos.
    """
    if df is None or df.empty:
        st.session_state["df"] = None
        st.session_state["_match_summary"] = None
        return
    d = deduplicate_docs(df) if dedup else df.copy()
    st.session_state["df"] = d
    st.session_state["_facturas_count"] = len(d)
    st.session_state["_match_summary"] = compute_match_summary(d)
    st.session_state["_match_timestamp"] = pd.Timestamp.utcnow().isoformat()

def reset_docs():
    """Elimina documentos cargados (crudos y normalizados)."""
    for k in ["df_raw","df","col_map","map_ok","_match_summary","facturas_summary"]:
        if k in st.session_state: st.session_state[k] = None
    if '_facturas_count' in st.session_state:
        st.session_state['_facturas_count'] = 0
    if '_match_timestamp' in st.session_state:
        st.session_state['_match_timestamp'] = None
    if 'df_cache' in st.session_state:
        st.session_state['df_cache'] = None
    if '_match_timestamp_view' in st.session_state:
        st.session_state['_match_timestamp_view'] = None

def reset_masters():
    """Elimina maestras (prioritarios y cuentas especiales)."""
    reset_proveedores()
    reset_cuentas_especiales()

# ============================================================
# 9) Resumen de match (para tarjetas de UI)
# ============================================================
def compute_match_summary(df: pd.DataFrame) -> dict:
    """Devuelve dict con conteos/porcentajes de match (cuenta especial / prioritario)."""
    if df is None or df.empty:
        return {"total": 0, "cuenta_especial": {"si":0,"no":0,"pct_si":0.0},
                "prov_prioritario": {"si":0,"no":0,"pct_si":0.0}}
    total = len(df)
    def _block(col):
        si = int(df[col].fillna(False).astype(bool).sum())
        no = total - si
        pct = (si / total * 100.0) if total else 0.0
        return {"si": si, "no": no, "pct_si": pct}
    return {
        "total": total,
        "cuenta_especial": _block("cuenta_especial"),
        "prov_prioritario": _block("prov_prioritario"),
    }

def get_match_summary() -> dict:
    """Último resumen almacenado o lo recalcula desde 'df'."""
    ss = st.session_state
    if ss.get("_match_summary") is not None:
        return ss["_match_summary"]
    d = ss.get("df")
    res = compute_match_summary(d if isinstance(d, pd.DataFrame) else None)
    ss["_match_summary"] = res
    return res

# ============================================================
# 10) Filtros (globales y avanzados)
# ============================================================
def _safe_minmax_date(df: pd.DataFrame, col: str):
    if col not in df:
        today = date.today()
        return date(today.year,1,1), today
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        today = date.today()
        return date(today.year,1,1), today
    return s.min().date(), s.max().date()

def general_date_filters_ui(df: pd.DataFrame):
    ss = st.session_state
    fac_min, fac_max = _safe_minmax_date(df, "fac_fecha_factura")
    pay_min, pay_max = _safe_minmax_date(df, "fecha_pagado")
    if ss.get("fac_ini") is None:
        ss["fac_ini"], ss["fac_fin"] = fac_min, fac_max
    if ss.get("pay_ini") is None:
        ss["pay_ini"], ss["pay_fin"] = pay_min, pay_max

    st.subheader("Filtros Globales")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Fecha de Factura")
        ss["fac_ini"] = st.date_input("Desde", value=ss["fac_ini"], key="fac_ini_ui", format="DD/MM/YYYY")
        ss["fac_fin"] = st.date_input("Hasta", value=ss["fac_fin"], key="fac_fin_ui", format="DD/MM/YYYY")
    with c2:
        st.markdown("##### Fecha de Pago")
        ss["pay_ini"] = st.date_input("Desde", value=ss["pay_ini"], key="pay_ini_ui", format="DD/MM/YYYY")
        ss["pay_fin"] = st.date_input("Hasta", value=ss["pay_fin"], key="pay_fin_ui", format="DD/MM/YYYY")
    return ss["fac_ini"], ss["fac_fin"], ss["pay_ini"], ss["pay_fin"]

def apply_general_filters(df: pd.DataFrame, fac_ini, fac_fin, pay_ini, pay_fin)->pd.DataFrame:
    out = df.copy()
    if "fac_fecha_factura" in out:
        out = out[(out["fac_fecha_factura"].dt.date >= fac_ini) & (out["fac_fecha_factura"].dt.date <= fac_fin)]
    if "fecha_pagado" in out:
        mask_pag = out["fecha_pagado"].isna() | (
            (out["fecha_pagado"].dt.date >= pay_ini) & (out["fecha_pagado"].dt.date <= pay_fin)
        )
        out = out[mask_pag]
    return out

def advanced_filters_ui(df: pd.DataFrame, labels: dict|None=None, helps: dict|None=None, show_controls=None, **_):
    """
    Devuelve (sede, org, prov, cc, oc, est, prio)
    (El prio global se maneja localmente en cada página, por eso lo devolvemos vacío)
    """
    ss = st.session_state
    show_controls = show_controls or ["sede","org","prov","cc","oc","est"]

    def _opts(df_, col):
        return sorted(df_[col].dropna().astype(str).unique().tolist()) if col in df_ else []

    opts_sede = _opts(df, "cmp_nombre")
    opts_org  = _opts(df, "origen")
    opts_prov = _opts(df, "prr_razon_social")
    opts_cc   = _opts(df, "nombre_centro_costo")

    oc_display = []
    if "con_oc" in df.columns:
        raw_oc_vals = sorted([int(v) for v in pd.to_numeric(df["con_oc"], errors="coerce").dropna().unique()])
        map_label = {1:"Con OC", 0:"Sin OC"}
        oc_display = [map_label.get(v, str(v)) for v in raw_oc_vals]

    est_raw = _opts(df, "estado_pago")
    est_display = [ESTADO_LABEL.get(x, x) for x in est_raw]

    with st.expander("▼ Filtros Avanzados"):
        c = st.columns(6)
        ss["sede_sel"] = c[0].multiselect("Sede", opts_sede, default=ss.get("sede_sel", [])) if "sede" in show_controls else []
        ss["org_sel"]  = c[1].multiselect("Origen", opts_org, default=ss.get("org_sel", [])) if "org" in show_controls else []
        ss["prov_sel"] = c[2].multiselect("Proveedor", opts_prov, default=ss.get("prov_sel", [])) if "prov" in show_controls else []
        ss["cc_sel"]   = c[3].multiselect("Centro de Costo", opts_cc, default=ss.get("cc_sel", [])) if "cc" in show_controls else []
        if "oc" in show_controls:
            default_oc = ss.get("oc_sel", [])
            ss["oc_sel"] = c[4].multiselect("Con OC", oc_display, default=default_oc)
        else:
            ss["oc_sel"] = []
        if "est" in show_controls:
            default_legibles = [ESTADO_LABEL.get(x, x) for x in ss.get("est_sel", [])]
            sel_legibles = c[5].multiselect("Tipo de Doc./Estado", est_display, default=default_legibles)
            ss["est_sel"] = [_INV_ESTADO_LABEL.get(x, x) for x in sel_legibles]
        else:
            ss["est_sel"] = []

    ss["prio_sel"] = []
    return ss["sede_sel"], ss["org_sel"], ss["prov_sel"], ss["cc_sel"], ss["oc_sel"], ss["est_sel"], ss["prio_sel"]

def apply_advanced_filters(df: pd.DataFrame, sede, org, prov, cc, oc, est, prio)->pd.DataFrame:
    out = df.copy()

    def m(col, vals):
        if not vals or col not in out: return out
        return out[out[col].astype(str).isin(list(map(str, vals)))]

    out = m("cmp_nombre", sede)
    out = m("origen", org)
    out = m("prr_razon_social", prov)
    out = m("nombre_centro_costo", cc)

    if oc and "con_oc" in out.columns:
        inv_map = {"Con OC": 1, "Sin OC": 0}
        oc_raw = [inv_map.get(x, x) for x in oc]
        out = out[out["con_oc"].isin(oc_raw)]

    out = m("estado_pago", est)
    return out
# ——— Filtro “chip” de Sede (auto-descubre columna sede / sede_pago) ———

def _detect_sede_col(df: pd.DataFrame) -> str | None:
    for c in ["sede", "sede_pago", "cmp_nombre"]:
        if c in df.columns:
            return c
    return None

def sede_chip_ui(df: pd.DataFrame, label: str = "Sede", key: str = "sede_chip"):
    """
    Muestra un selector horizontal estilo chip: 'Todas', <sede1>, <sede2>, ...
    Devuelve:
      - None  -> si el usuario elige 'Todas'
      - str   -> nombre de la sede seleccionada
    """
    col = _detect_sede_col(df)
    if not col:
        st.caption("⚠️ No se encontró columna de sede (ni 'sede' ni 'sede_pago').")
        return None

    opciones = (
        df[col]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s != ""]
        .sort_values()
        .unique()
        .tolist()
    )
    opciones = ["Todas"] + opciones

    # Si tu Streamlit es reciente, puedes usar st.segmented_control; si no, usamos radio horizontal
    try:
        selected = st.segmented_control(label, opciones, key=key)  # Streamlit ≥ 1.36
    except Exception:
        selected = st.radio(label, opciones, horizontal=True, key=key)  # fallback universal

    return None if selected == "Todas" else selected

def apply_sede_chip(df: pd.DataFrame, sede_sel: str | None) -> pd.DataFrame:
    """Aplica el filtro de sede si corresponde; si sede_sel es None, no filtra."""
    col = _detect_sede_col(df)
    if not col or sede_sel is None:
        return df
    return df[df[col].astype(str).str.strip() == str(sede_sel).strip()]
