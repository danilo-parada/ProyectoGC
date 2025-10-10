from __future__ import annotations

from typing import Dict, Mapping, Sequence, Any, Optional

import logging

import numpy as np
import pandas as pd

from lib_common import apply_advanced_filters


def compute_monto_pagado_real(df: pd.DataFrame) -> pd.Series:
    f = df.copy()
    index = f.index

    a_raw = f.get("monto_autorizado")
    if isinstance(a_raw, pd.Series):
        a = pd.to_numeric(a_raw, errors="coerce")
    else:
        a = pd.Series(a_raw, index=index)
        a = pd.to_numeric(a, errors="coerce")

    p_raw = f.get("monto_pagado")
    if isinstance(p_raw, pd.Series):
        p = pd.to_numeric(p_raw, errors="coerce")
    else:
        p = pd.Series(p_raw, index=index)
        p = pd.to_numeric(p, errors="coerce")

    fecha_raw = f.get("fecha_pagado")
    if isinstance(fecha_raw, pd.Series):
        fecha = pd.to_datetime(fecha_raw, errors="coerce")
    else:
        fecha = pd.Series(fecha_raw, index=index)
        fecha = pd.to_datetime(fecha, errors="coerce")

    p = p.fillna(0)
    a = a.fillna(np.nan)

    cond1 = fecha.notna() & (p > 0)
    cond2 = fecha.notna() & (p <= 0) & (a.fillna(0) > 0)
    cond3 = fecha.isna() & (p > 0)

    out = pd.Series(0.0, index=index, dtype="float64")
    out.loc[cond1] = a.where(a.notna(), p).loc[cond1]
    out.loc[cond2] = a.fillna(0).loc[cond2]
    out.loc[cond3] = p.loc[cond3]
    return out.clip(lower=0)

def _safe_to_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    try:
        out = pd.to_numeric(s, errors="coerce")
        return out.fillna(default)
    except Exception:
        return pd.Series([default] * len(s), index=s.index)

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.NaT)

def ensure_derived_fields(df_in: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Crea columnas derivadas comunes sin modificar ``df_in``.

    Cuando la entrada es ``None`` se retorna un ``DataFrame`` vacío para que
    los consumidores puedan continuar con una estructura válida sin errores de
    atributo. Esto ocurre, por ejemplo, en las páginas de Streamlit antes de que
    el usuario cargue información.
    """

    if df_in is None:
        return pd.DataFrame()

    if not isinstance(df_in, pd.DataFrame):
        try:
            df = pd.DataFrame(df_in)
        except Exception:
            return pd.DataFrame()
    else:
        df = df_in.copy()

    # Fechas y montos
    for c in ["fac_fecha_factura", "fecha_autoriza", "fecha_pagado", "fecha_cc"]:
        if c in df.columns: df[c] = _safe_to_datetime(df[c])
    for c in ["fac_monto_total", "monto_autorizado", "monto_pagado", "monto_ce", "con_oc"]:
        if c in df.columns:
            df[c] = _safe_to_numeric(df[c], default=0.0)

    # Columnas estándar de pago
    if "monto_ce" not in df.columns:
        df["monto_ce"] = _safe_to_numeric(df.get("monto_pagado", pd.Series(0, index=df.index)), default=0.0)
    else:
        df["monto_ce"] = _safe_to_numeric(df["monto_ce"], default=0.0)

    if "fecha_ce" not in df.columns:
        fallback = df.get("fecha_pago_ce")
        if fallback is None:
            fallback = df.get("fecha_pagado")
        df["fecha_ce"] = _safe_to_datetime(fallback if fallback is not None else pd.Series(pd.NaT, index=df.index))
    else:
        df["fecha_ce"] = _safe_to_datetime(df["fecha_ce"])

    # Estado pago si falta
    maut = df.get("monto_autorizado", pd.Series(0, index=df.index)).fillna(0.0)
    monto_ce = df.get("monto_ce", pd.Series(0, index=df.index)).fillna(0.0)
    fecha_ce = pd.to_datetime(df.get("fecha_ce"), errors="coerce") if "fecha_ce" in df.columns else pd.Series(pd.NaT, index=df.index)
    estado_pagada = (monto_ce > 0) | fecha_ce.notna()
    df["is_pagada"] = estado_pagada

    if "estado_pago" not in df.columns:
        df["estado_pago"] = np.select(
            [estado_pagada, (maut > 0) & ~estado_pagada],
            ["pagada", "autorizada_sin_pago"],
            default="sin_autorizacion",
        )
    else:
        df.loc[estado_pagada, "estado_pago"] = "pagada"

    # Derivadas de tiempo
    if "fac_fecha_factura" in df.columns and "fecha_autoriza" in df.columns:
        df["dias_factura_autorizacion"] = (df["fecha_autoriza"] - df["fac_fecha_factura"]).dt.days
    if "fac_fecha_factura" in df.columns and "fecha_pagado" in df.columns:
        df["dias_a_pago_calc"] = (df["fecha_pagado"] - df["fac_fecha_factura"]).dt.days
    if "fecha_autoriza" in df.columns and "fecha_pagado" in df.columns:
        df["dias_autorizacion_pago_calc"] = (df["fecha_pagado"] - df["fecha_autoriza"]).dt.days

    # Transcurridos / deuda
    hoy = pd.Timestamp.today().normalize()
    if "fac_fecha_factura" in df.columns:
        df["dias_transcurridos_estado"] = np.where(
            df["estado_pago"].eq("pagada"),
            df.get("dias_a_pago_calc", pd.Series(np.nan, index=df.index)),
            (hoy - df["fac_fecha_factura"]).dt.days,
        )
        if "fecha_venc_30" not in df.columns:
            df["fecha_venc_30"] = df["fac_fecha_factura"] + pd.to_timedelta(30, unit="D")
        if "dias_a_vencer" not in df.columns:
            df["dias_a_vencer"] = (df["fecha_venc_30"] - hoy).dt.days
        if "importe_deuda" not in df.columns:
            df["importe_deuda"] = np.where(
                df["estado_pago"].eq("autorizada_sin_pago"),
                df.get("monto_autorizado", 0.0),
                df.get("fac_monto_total", 0.0),
            )
        if "Nivel" not in df.columns:
            df["Nivel"] = df["estado_pago"].map({
                "autorizada_sin_pago": "Contabilizado Pendiente de Pago",
                "sin_autorizacion": "Pendiente de Contabilización",
            })

    if "prov_prioritario" not in df.columns:
        df["prov_prioritario"] = False

    return df

def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _sum_numeric(series: Optional[pd.Series]) -> float:
    if series is None:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float(values.sum())


def _paid_mask(
    df: pd.DataFrame,
    amount_col: Optional[str],
    pay_date_col: Optional[str],
) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    if "estado_pago" in df.columns:
        estados = df["estado_pago"].astype(str).str.lower()
        mask = mask | estados.str.contains("pagad", na=False)

    if amount_col and amount_col in df.columns:
        monto = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
        mask = mask | (monto > 0)

    if pay_date_col and pay_date_col in df.columns:
        fechas_pago = pd.to_datetime(df[pay_date_col], errors="coerce")
        mask = mask | fechas_pago.notna()

    return mask


def _mean_days(
    df: pd.DataFrame,
    end_col: Optional[str],
    start_col: Optional[str],
    *,
    clip: tuple[int, int] = (-5, 365),
) -> float:
    if not end_col or not start_col:
        return float("nan")

    fechas_fin = pd.to_datetime(df[end_col], errors="coerce")
    fechas_ini = pd.to_datetime(df[start_col], errors="coerce")
    delta = (fechas_fin - fechas_ini).dt.days
    delta = delta.dropna()
    if clip:
        delta = delta[(delta >= clip[0]) & (delta <= clip[1])]
    if delta.empty:
        return float("nan")
    return float(np.round(delta.mean(), 1))


def apply_common_filters(df: pd.DataFrame, filtros: Mapping[str, Any]) -> pd.DataFrame:
    """Aplica filtros compartidos para Tablero KPI e Informe Asesor."""

    if df is None:
        return pd.DataFrame()

    base = ensure_derived_fields(df)

    fac_range = filtros.get("fac_range")
    pay_range = filtros.get("pay_range")

    if fac_range and len(fac_range) == 2 and all(fac_range):
        fac_ini, fac_fin = fac_range
        if "fac_fecha_factura" in base.columns:
            fechas_fac = pd.to_datetime(base["fac_fecha_factura"], errors="coerce")
            mask_fac = fechas_fac.dt.date.between(fac_ini, fac_fin)
            base = base[mask_fac]

    if pay_range and len(pay_range) == 2 and all(pay_range):
        pay_ini, pay_fin = pay_range
        if "fecha_pagado" in base.columns:
            fechas_pago = pd.to_datetime(base["fecha_pagado"], errors="coerce")
            mask_pay = fechas_pago.isna() | fechas_pago.dt.date.between(pay_ini, pay_fin)
            base = base[mask_pay]

    sede = filtros.get("sede", [])
    org = filtros.get("org", [])
    prov = filtros.get("prov", [])
    cc = filtros.get("cc", [])
    oc = filtros.get("oc", [])
    est = filtros.get("est", [])
    estado_doc = filtros.get("estado_doc")
    prio = filtros.get("prio", [])

    base = apply_advanced_filters(base, sede, org, prov, cc, oc, est, prio)
    if estado_doc not in (None, "", "Todos") and "estado_doc" in base.columns:
        estado_norm = base["estado_doc"].astype(str)
        base = base[estado_norm == str(estado_doc)]
    return base


def _sanitize_monto(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _sanitize_datetime(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce")


def is_pagada_row(row: pd.Series) -> bool:
    """Determina si una fila representa un documento pagado según reglas estándar."""

    monto = pd.to_numeric(pd.Series([row.get("monto_ce", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
    fecha = pd.to_datetime(pd.Series([row.get("fecha_ce")]), errors="coerce").iloc[0]
    return bool((monto > 0) or pd.notna(fecha))


def _derive_pagadas(df: pd.DataFrame) -> pd.Series:
    monto_ce = _sanitize_monto(df.get("monto_ce"))
    fecha_ce = _sanitize_datetime(df.get("fecha_ce"))
    return monto_ce.gt(0) | fecha_ce.notna()


def _mean_days_clip(
    df: pd.DataFrame,
    start_col: Optional[str],
    end_col: Optional[str],
    *,
    mask: Optional[pd.Series] = None,
    clip: tuple[int, int] = (-5, 365),
) -> float:
    if not start_col or not end_col or start_col not in df.columns or end_col not in df.columns:
        return float("nan")

    dates_start = _sanitize_datetime(df[start_col])
    dates_end = _sanitize_datetime(df[end_col])
    valid_mask = dates_start.notna() & dates_end.notna()
    if mask is not None:
        valid_mask &= mask

    if not valid_mask.any():
        return float("nan")

    delta = (dates_end - dates_start).dt.days.loc[valid_mask]
    if clip:
        delta = delta[(delta >= clip[0]) & (delta <= clip[1])]
    if delta.empty:
        return float("nan")
    return float(np.round(delta.mean(), 1))


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula KPIs monetarios y de tiempos sobre un DataFrame ya filtrado."""

    if df is None or df.empty:
        return {
            "total_facturado": 0.0,
            "total_pagado": 0.0,
            "total_pagado_real": 0.0,
            "facturado_pagado": 0.0,
            "facturado_sin_pagar": 0.0,
            "dpp": float("nan"),
            "dic": float("nan"),
            "dcp": float("nan"),
            "brecha_pct": 0.0,
            "docs_total": 0.0,
            "docs_pagados": 0.0,
        }

    data = df.copy()

    data["monto_pagado_real"] = compute_monto_pagado_real(data)
    data["monto_facturado"] = _sanitize_monto(
        data.get("monto_facturado", data.get("fac_monto_total", 0.0))
    )

    data["fecha_emision"] = _sanitize_datetime(data.get("fecha_emision", data.get("fac_fecha_factura")))
    data["fecha_contab"] = _sanitize_datetime(data.get("fecha_contabilizacion", data.get("fecha_cc")))
    data["fecha_pagado"] = _sanitize_datetime(data.get("fecha_pagado"))

    pagadas_monto = data["monto_pagado_real"] > 0

    total_facturado = float(data["monto_facturado"].sum())
    total_pagado_real = float(data["monto_pagado_real"].sum())
    facturado_pagado = float(data.loc[pagadas_monto, "monto_facturado"].sum())
    facturado_sin_pagar = float(data.loc[~pagadas_monto, "monto_facturado"].sum())

    brecha_pct = 100.0 * (total_pagado_real - total_facturado) / max(total_facturado, 1.0)

    pagadas_fecha = data["fecha_pagado"].notna()
    dpp = _mean_days_clip(data, "fecha_emision", "fecha_pagado", mask=pagadas_fecha)
    dic = _mean_days_clip(data, "fecha_emision", "fecha_contab")
    dcp = _mean_days_clip(data, "fecha_contab", "fecha_pagado", mask=pagadas_fecha)

    if abs((facturado_pagado + facturado_sin_pagar) - total_facturado) > 0.51:
        logging.getLogger(__name__).warning("Desbalance en desglose de facturación detectado durante compute_kpis")

    return {
        "total_facturado": total_facturado,
        "total_pagado": total_pagado_real,
        "total_pagado_real": total_pagado_real,
        "facturado_pagado": facturado_pagado,
        "facturado_sin_pagar": facturado_sin_pagar,
        "dpp": dpp,
        "dic": dic,
        "dcp": dcp,
        "brecha_pct": brecha_pct,
        "docs_total": float(len(data)),
        "docs_pagados": float(pagadas_monto.sum()),
    }


def _dic_mean(delta: pd.Series) -> float:
    if delta is None or delta.empty:
        return float("nan")

    clean = delta.dropna()
    clean = clean[(clean >= -5) & (clean <= 365)]
    if clean.empty:
        return float("nan")

    return float(np.round(clean.mean(), 1))


def compute_dic_split(df: pd.DataFrame) -> Dict[str, float]:
    """Obtiene promedios DIC segmentados por estado de pago."""

    if df is None or df.empty:
        return {
            "dic_pagadas_avg": float("nan"),
            "dic_pagadas_n": 0,
            "dic_contab_unpaid_avg": float("nan"),
            "dic_contab_unpaid_n": 0,
            "no_contab_n": 0,
        }

    data = ensure_derived_fields(df)

    if "monto_pagado_real" not in data.columns:
        data["monto_pagado_real"] = compute_monto_pagado_real(data)

    fecha_emision = _sanitize_datetime(data.get("fecha_emision", data.get("fac_fecha_factura")))
    fecha_contab = _sanitize_datetime(data.get("fecha_contabilizacion", data.get("fecha_cc")))

    dic_delta = (fecha_contab - fecha_emision).dt.days

    pagadas_mask = _derive_pagadas(data)
    contab_mask = fecha_contab.notna() & fecha_emision.notna()

    pagadas_valid = pagadas_mask & contab_mask
    unpaid_valid = (~pagadas_mask) & contab_mask
    no_contab_mask = fecha_contab.isna()

    dic_pagadas_avg = _dic_mean(dic_delta.loc[pagadas_valid])
    dic_contab_unpaid_avg = _dic_mean(dic_delta.loc[unpaid_valid])

    return {
        "dic_pagadas_avg": dic_pagadas_avg,
        "dic_pagadas_n": int(pagadas_valid.sum()),
        "dic_contab_unpaid_avg": dic_contab_unpaid_avg,
        "dic_contab_unpaid_n": int(unpaid_valid.sum()),
        "no_contab_n": int(no_contab_mask.sum()),
    }


def prepare_hist_data(df: pd.DataFrame, column: str, max_days: int = 100) -> pd.DataFrame:
    """Dataset limpio para histogramas (sin negativos/NaN, recorte por ventana)."""
    d = ensure_derived_fields(df)
    if column not in d.columns:
        return pd.DataFrame(columns=[column])
    s = d[column].dropna()
    s = s[s >= 0]
    s = s[s <= max_days]
    return pd.DataFrame({column: s})
