from __future__ import annotations

from typing import Dict, Mapping, Sequence, Any

import numpy as np
import pandas as pd

from lib_common import apply_advanced_filters

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

def ensure_derived_fields(df_in: pd.DataFrame | None) -> pd.DataFrame:
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
    for c in ["fac_monto_total", "monto_autorizado", "monto_pagado", "con_oc"]:
        if c in df.columns: df[c] = _safe_to_numeric(df[c], default=0.0)

    # Estado pago si falta
    if "estado_pago" not in df.columns:
        mpag = df.get("monto_pagado", pd.Series(0, index=df.index)).fillna(0)
        maut = df.get("monto_autorizado", pd.Series(0, index=df.index)).fillna(0)
        df["estado_pago"] = np.select(
            [mpag > 0, (maut > 0) & (mpag == 0), maut == 0],
            ["pagada", "autorizada_sin_pago", "sin_autorizacion"],
            default="sin_autorizacion",
        )

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
                "autorizada_sin_pago": "Doc. Autorizado p/ Pago",
                "sin_autorizacion": "Doc. Pendiente de Autorización",
            })

    if "prov_prioritario" not in df.columns:
        df["prov_prioritario"] = False

    return df

def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _sum_numeric(series: pd.Series | None) -> float:
    if series is None:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float(values.sum())


def _paid_mask(df: pd.DataFrame, amount_col: str | None, pay_date_col: str | None) -> pd.Series:
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
    end_col: str | None,
    start_col: str | None,
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
    prio = filtros.get("prio", [])

    base = apply_advanced_filters(base, sede, org, prov, cc, oc, est, prio)
    return base


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula KPIs monetarios y de tiempos sobre un DataFrame ya filtrado."""

    if df is None or df.empty:
        return {
            "total_facturado": 0.0,
            "total_pagado": 0.0,
            "dpp": float("nan"),
            "dic": float("nan"),
            "dcp": float("nan"),
            "brecha_pct": 0.0,
            "docs_total": 0.0,
            "docs_pagados": 0.0,
        }

    data = df.copy()

    for col in [
        "fecha_emision",
        "fac_fecha_factura",
        "fecha_pago_ce",
        "fecha_pagado",
        "fecha_contabilizacion",
        "fecha_cc",
    ]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    fact_col = _first_present(data, ["monto_facturado", "fac_monto_total"])
    pago_col = _first_present(data, ["monto_pagado"])
    fecha_emision_col = _first_present(data, ["fecha_emision", "fac_fecha_factura"])
    fecha_pago_col = _first_present(data, ["fecha_pago_ce", "fecha_pagado"])
    fecha_contab_col = _first_present(data, ["fecha_contabilizacion", "fecha_cc"])

    total_facturado = _sum_numeric(data.get(fact_col)) if fact_col else 0.0

    paid_mask = _paid_mask(data, pago_col, fecha_pago_col)
    data_pagadas = data[paid_mask].copy()
    total_pagado = _sum_numeric(data_pagadas.get(pago_col)) if pago_col else 0.0

    dpp = _mean_days(data_pagadas, fecha_pago_col, fecha_emision_col)
    dic = _mean_days(data, fecha_contab_col, fecha_emision_col)
    dcp = _mean_days(data, fecha_pago_col, fecha_contab_col)

    brecha_pct = 100.0 * (total_pagado - total_facturado) / max(total_facturado, 1.0)

    return {
        "total_facturado": total_facturado,
        "total_pagado": total_pagado,
        "dpp": dpp,
        "dic": dic,
        "dcp": dcp,
        "brecha_pct": brecha_pct,
        "docs_total": float(len(data)),
        "docs_pagados": float(paid_mask.sum()),
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
