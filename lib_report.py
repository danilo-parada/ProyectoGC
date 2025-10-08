from __future__ import annotations
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape

from core.utils import LABELS
from lib_common import money, one_decimal


# ===================== Excel helpers =====================
def excel_bytes_single(df: pd.DataFrame, sheet_name: str = "Datos") -> bytes:
    buf = BytesIO()
    safe = (sheet_name or "Datos")[:31]
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        (df if isinstance(df, pd.DataFrame) and not df.empty
         else pd.DataFrame([["Sin datos"]], columns=["Aviso"])
        ).to_excel(writer, index=False, sheet_name=safe)
    buf.seek(0)
    return buf.getvalue()


def excel_bytes_multi(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        if not sheets:
            pd.DataFrame([["Sin datos"]], columns=["Aviso"]).to_excel(
                writer, index=False, sheet_name="Resumen"
            )
        else:
            for name, df in sheets.items():
                safe = (name or "Hoja")[:31]
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(writer, index=False, sheet_name=safe)
                else:
                    pd.DataFrame([["Sin datos"]], columns=["Aviso"]).to_excel(
                        writer, index=False, sheet_name=safe
                    )
    buf.seek(0)
    return buf.getvalue()


# ===================== PDF styles/helpers =====================
def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="h1_center", parent=styles["h1"], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="body_left", parent=styles["BodyText"],
                              alignment=TA_LEFT, fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="cell_left", parent=styles["BodyText"],
                              alignment=TA_LEFT, fontSize=8, leading=10, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="cell_center", parent=styles["BodyText"],
                              alignment=TA_CENTER, fontSize=8, leading=10, wordWrap="CJK"))
    return styles


def _auto_col_widths_compact(columns: List[str], available_width: float) -> List[float]:
    weights: Dict[str, float] = {}
    for col in columns:
        c = col.lower()
        if c in ("proveedor",):
            weights[col] = 3.2
        elif c in ("sede", "cuenta corriente", "banco"):
            weights[col] = 1.7
        elif c in ("n° factura", "dias v", "días v", "pp", "ce"):
            weights[col] = 0.9
        elif "monto" in c:
            weights[col] = 1.4
        elif "fecha" in c:
            weights[col] = 1.4
        elif c == "nivel":
            weights[col] = 1.2
        else:
            weights[col] = 1.0
    total = sum(weights.values()) or 1.0
    return [available_width * (weights[c] / total) for c in columns]


def _is_scalar(x) -> bool:
    return np.isscalar(x) or isinstance(x, (pd.Timestamp,))


def _wrap_dataframe(df: pd.DataFrame, styles, left_align_cols: List[str]) -> List[List]:
    cols = df.columns.tolist()
    data = [cols]
    for _, row in df.iterrows():
        rendered = []
        for c in cols:
            val = row[c]
            if not _is_scalar(val):
                text = str(val)
            else:
                text = "" if pd.isna(val) else str(val)
            style = styles["cell_left"] if c in left_align_cols else styles["cell_center"]
            rendered.append(Paragraph(text, style))
        data.append(rendered)
    return data


def _chunk_rows(df: pd.DataFrame, size: int = 40) -> List[pd.DataFrame]:
    if df.empty:
        return [df]
    return [df.iloc[i:i+size].copy() for i in range(0, len(df), size)]


def _limit_columns(df: pd.DataFrame, max_cols: int = 12) -> pd.DataFrame:
    if df.shape[1] <= max_cols:
        return df
    return df.iloc[:, :max_cols].copy()


# ===================== Helpers robustos =====================
def _safe_numeric_series(d: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """
    Devuelve una Serie numérica (mismo índice de d) desde d[col].
    - Si hay columnas duplicadas (d[col] -> DataFrame), usa la primera.
    - Si no existe la columna, devuelve Serie llena con default.
    - Convierte a float con errors='coerce' y fillna(default).
    """
    if col in d.columns:
        ser = d[col]
        # Si hay duplicados, pandas devuelve DataFrame:
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        if not isinstance(ser, pd.Series):
            # Último recurso: repetir el valor en todo el índice
            ser = pd.Series([ser] * len(d), index=d.index)
        ser = pd.to_numeric(ser, errors="coerce").fillna(default)
    else:
        ser = pd.Series(default, index=d.index, dtype=float)
    return ser


# ===================== Compact schema =====================
_COMPACT_ORDER = [
    "Nivel", "PP", "CE", "N° Factura", "Sede", "Proveedor",
    "Fecha Venc.", "Días V", "Monto", "Cuenta Corriente", "Banco"
]


def _to_compact_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_COMPACT_ORDER)

    d = df.copy()

    # Nivel
    if "Nivel" not in d.columns:
        d["Nivel"] = ""
    d["Nivel"] = d["Nivel"].replace({
        "Doc. Autorizado p/ Pago": "Aut.p/Pago",
        "Doc. Pendiente de Autorización": "Pen. Aut."
    }).fillna("")

    # PP y CE -> Sí/NO (si faltan, usar False)
    if "prov_prioritario" not in d.columns:
        d["prov_prioritario"] = False
    if "cuenta_especial" not in d.columns:
        d["cuenta_especial"] = False

    d["PP"] = d["prov_prioritario"].map({True: "Sí", False: "NO"})
    d["CE"] = d["cuenta_especial"].map({True: "Sí", False: "NO"})

    # N° Factura, Sede, Proveedor
    if "N° Factura" not in d.columns and "fac_numero" in d.columns:
        d["N° Factura"] = d["fac_numero"]
    if "Sede" not in d.columns and "cmp_nombre" in d.columns:
        d["Sede"] = d["cmp_nombre"]
    if "Proveedor" not in d.columns and "prr_razon_social" in d.columns:
        d["Proveedor"] = d["prr_razon_social"]

    # Fecha Venc.
    if "Fecha Venc." not in d.columns:
        if "fecha_venc_30" in d.columns:
            d["Fecha Venc."] = d["fecha_venc_30"]
        else:
            d["Fecha Venc."] = pd.NaT

    # Días V
    if "Días V" not in d.columns:
        if "dias_a_vencer" in d.columns:
            d["Días V"] = _safe_numeric_series(d, "dias_a_vencer", default=np.nan)
        else:
            d["Días V"] = pd.NA

    # Monto (importe_deuda o importe_regla; si no existe → Serie de ceros)
    if "Monto" not in d.columns:
        if "importe_deuda" in d.columns:
            ser = _safe_numeric_series(d, "importe_deuda", default=0.0)
        elif "importe_regla" in d.columns:
            ser = _safe_numeric_series(d, "importe_regla", default=0.0)
        else:
            ser = pd.Series(0.0, index=d.index, dtype=float)
        d["Monto"] = ser

    # Cuenta Corriente / Banco
    if "Cuenta Corriente" not in d.columns and "cuenta_corriente" in d.columns:
        d["Cuenta Corriente"] = d["cuenta_corriente"]
    if "Banco" not in d.columns and "banco" in d.columns:
        d["Banco"] = d["banco"]

    # Selección y orden final
    keep = [c for c in _COMPACT_ORDER if c in d.columns]
    out = d[keep].copy()

    # Tipos
    if "Monto" in out.columns:
        out["Monto"] = _safe_numeric_series(out, "Monto", default=0.0)
    if "Días V" in out.columns:
        out["Días V"] = pd.to_numeric(out["Días V"], errors="coerce")

    return out


# ===================== Table builder (compact) =====================
def _table_compact(
    df: pd.DataFrame,
    available_width: float,
    header_bg,
    body_bg,
    header_text=colors.whitesmoke,
    grid_color=colors.black,
    styles=None,
) -> Table:
    styles = styles or _styles()
    df2 = df.copy()

    # Formato dinero seguro
    if "Monto" in df2.columns:
        try:
            ser_num = pd.to_numeric(df2["Monto"], errors="coerce")
            if isinstance(ser_num, pd.Series) and ser_num.notna().any():
                df2["Monto"] = ser_num.fillna(0).map(money)
        except Exception:
            pass

    # Fecha a texto corto
    if "Fecha Venc." in df2.columns:
        try:
            df2["Fecha Venc."] = pd.to_datetime(df2["Fecha Venc."], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            df2["Fecha Venc."] = df2["Fecha Venc."].astype(str)

    columns = df2.columns.tolist()
    col_widths = _auto_col_widths_compact(columns, available_width)
    left_align = [c for c in columns if c in ("Proveedor", "Sede", "Banco", "Cuenta Corriente")]

    table_data = _wrap_dataframe(df2, styles, left_align_cols=left_align)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_text),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),

        ("BACKGROUND", (0, 1), (-1, -1), body_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, grid_color),

        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return tbl


def _safe_plot_image(fig: go.Figure, max_w: float, max_h: float) -> Optional[Image]:
    if not isinstance(fig, go.Figure):
        return None
    try:
        buf = BytesIO()
        fig.write_image(buf, format="png", width=1200, height=600, scale=2)
        buf.seek(0)
        return Image(buf, width=max_w, height=max_h, kind="proportional")
    except Exception:
        return None


# ===================== PDF principal =====================
def generate_pdf_report(
    secciones: Dict,
    kpis: Dict,
    rankings_df: Optional[pd.DataFrame] = None,
    niveles_kpis: Optional[Dict] = None,
    proyeccion_chart: Optional[go.Figure] = None,
    pagos_criticos_df: Optional[pd.DataFrame] = None,
    sugerencias: Optional[List[str]] = None,
    filtros: Optional[Dict] = None,
    presupuesto_monto: Optional[float] = None,
    seleccion_hoy_df: Optional[pd.DataFrame] = None,
) -> Optional[bytes]:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title="Informe Ejecutivo Financiero",
    )
    styles = _styles()
    story = []

    # Encabezado
    story.append(Paragraph("Informe Ejecutivo Financiero", styles["h1_center"]))
    story.append(Paragraph(f"Generado el: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", styles["body_left"]))
    story.append(Spacer(1, 0.12 * inch))

    # Filtros
    if filtros:
        story.append(Paragraph("<b>Filtros Aplicados</b>", styles["body_left"]))
        filtros_txt = (
            f"• Período Facturas: <b>{filtros.get('fac_ini','-')}</b> a <b>{filtros.get('fac_fin','-')}</b><br/>"
            f"• Período Pagos: <b>{filtros.get('pay_ini','-')}</b> a <b>{filtros.get('pay_fin','-')}</b>"
        )
        story.append(Paragraph(filtros_txt, styles["body_left"]))
        story.append(Spacer(1, 0.12 * inch))

    # KPIs
    if secciones.get("kpis"):
        story.append(Paragraph("<b>Resumen KPIs de Pagos (Facturas Pagadas)</b>", styles["body_left"]))
        kpi_txt = (
            f"• Monto Total Facturado: <b>{kpis.get('total_facturado','-')}</b><br/>"
            f"• Total Pagado (autorizado): <b>{kpis.get('total_pagado','-')}</b><br/>"
            f"• {LABELS['dpp_emision_pago']}: <b>{kpis.get('dso','-')}</b>"
        )
        story.append(Paragraph(kpi_txt, styles["body_left"]))
        story.append(Spacer(1, 0.12 * inch))

    # Rankings (si corresponde)
    if secciones.get("rankings") and isinstance(rankings_df, pd.DataFrame) and not rankings_df.empty:
        story.append(Paragraph("<b>Top Proveedores (por Monto Autorizado)</b>", styles["body_left"]))
        order_cols = [
            "Razón Social","Monto Autorizado","Monto Pagado",
            "Días Promedio Pago","Cant. Fact. ≤30 días","Cant. Fact. >30 días"
        ]
        df_rank = rankings_df.copy()
        keep = [c for c in order_cols if c in df_rank.columns] or list(df_rank.columns)[:6]
        df_rank = df_rank[keep]
        for c in ("Monto Autorizado","Monto Pagado"):
            if c in df_rank:
                df_rank[c] = pd.to_numeric(df_rank[c], errors="coerce").fillna(0)
        if "Días Promedio Pago" in df_rank:
            df_rank["Días Promedio Pago"] = pd.to_numeric(df_rank["Días Promedio Pago"], errors="coerce").apply(one_decimal)

        tbl_rank = _table_compact(
            df_rank, available_width=doc.width,
            header_bg=colors.darkblue, body_bg=colors.lightblue, styles=styles
        )
        story.append(tbl_rank)
        story.append(Spacer(1, 0.12 * inch))

    # Análisis de Deuda Pendiente
    if secciones.get("deuda"):
        story.append(Paragraph("<b>Análisis de Deuda Pendiente</b>", styles["body_left"]))
        story.append(Spacer(1, 0.08 * inch))

        if isinstance(proyeccion_chart, go.Figure):
            img = _safe_plot_image(proyeccion_chart, max_w=doc.width, max_h=3.3 * inch)
            if img:
                story.append(Paragraph("Gráfico de Vencimientos Futuros", styles["body_left"]))
                story.append(img)
                story.append(Spacer(1, 0.08 * inch))

        # Pagos Críticos (azul)
        if isinstance(pagos_criticos_df, pd.DataFrame) and not pagos_criticos_df.empty:
            story.append(Paragraph("<b>Pagos Críticos</b>", styles["body_left"]))
            df_c = _to_compact_schema(pagos_criticos_df)
            df_c = _limit_columns(df_c, 11)

            for chunk in _chunk_rows(df_c, 40):
                tbl_c = _table_compact(
                    chunk, available_width=doc.width,
                    header_bg=colors.darkblue, body_bg=colors.lightblue, styles=styles
                )
                story.append(tbl_c)
                story.append(Spacer(1, 0.08 * inch))

        # Selección a Pagar Hoy (rojo)
        if isinstance(seleccion_hoy_df, pd.DataFrame) and not seleccion_hoy_df.empty:
            story.append(Paragraph("<b>Selección a Pagar Hoy (CRÍTICO)</b>", styles["body_left"]))
            if presupuesto_monto is not None:
                story.append(Paragraph(f"Presupuesto: <b>{money(presupuesto_monto)}</b>", styles["body_left"]))
            df_sel = _to_compact_schema(seleccion_hoy_df)
            df_sel = _limit_columns(df_sel, 11)

            for chunk in _chunk_rows(df_sel, 40):
                tbl_sel = _table_compact(
                    chunk, available_width=doc.width,
                    header_bg=colors.HexColor("#B61D1D"),
                    body_bg=colors.HexColor("#FFEAEA"),
                    styles=styles
                )
                story.append(tbl_sel)
                story.append(Spacer(1, 0.08 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()