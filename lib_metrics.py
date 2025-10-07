from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict

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

def ensure_derived_fields(df_in: pd.DataFrame) -> pd.DataFrame:
    """Crea columnas derivadas comunes sin modificar df original."""
    df = df_in.copy()

    # Fechas y montos
    for c in ["fac_fecha_factura", "fecha_autoriza", "fecha_pagado"]:
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

def split_by_estado(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (pagadas, autorizadas_sin_pago, sin_autorizacion)."""
    d = df.copy()
    pag = d[d["estado_pago"] == "pagada"].copy()
    aut = d[d["estado_pago"] == "autorizada_sin_pago"].copy()
    sin = d[d["estado_pago"] == "sin_autorizacion"].copy()
    return pag, aut, sin

def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """KPIs comunes del período filtrado."""
    d = ensure_derived_fields(df)
    total_fact = float(d.get("fac_monto_total", pd.Series(dtype=float)).sum())
    pag, aut_sp, _ = split_by_estado(d)

    total_aut_pag = float(pag.get("monto_autorizado", pd.Series(dtype=float)).sum())
    monto_aut_sin_pago = float(aut_sp.get("monto_autorizado", pd.Series(dtype=float)).sum())
    cnt_pag, cnt_aut, cnt_sin = len(pag), len(aut_sp), len(d) - len(pag) - len(aut_sp)

    # DSO
    dso = 0.0
    if "dias_a_pago_calc" in pag.columns:
        v = pag.loc[pag["dias_a_pago_calc"] >= 0, "dias_a_pago_calc"].dropna()
        if not v.empty: dso = float(v.mean())

    # TFA
    tfa = 0.0
    if "dias_factura_autorizacion" in d.columns:
        v = d.loc[d["dias_factura_autorizacion"] >= 0, "dias_factura_autorizacion"].dropna()
        if not v.empty: tfa = float(v.mean())

    # TPA
    tpa = 0.0
    if "dias_autorizacion_pago_calc" in pag.columns:
        v = pag.loc[pag["dias_autorizacion_pago_calc"] >= 0, "dias_autorizacion_pago_calc"].dropna()
        if not v.empty: tpa = float(v.mean())

    total_fact_en_pagadas = float(pag.get("fac_monto_total", pd.Series(dtype=float)).sum())
    gap_pct = (1 - (total_aut_pag / total_fact_en_pagadas)) * 100 if total_fact_en_pagadas > 0 else 0.0

    return {
        "total_facturado": total_fact,
        "total_pagado_aut": total_aut_pag,
        "monto_aut_sin_pago": monto_aut_sin_pago,
        "cnt_pag": float(cnt_pag),
        "cnt_aut_sin_pago": float(cnt_aut),
        "cnt_sin_aut": float(cnt_sin),
        "dso": dso, "tfa": tfa, "tpa": tpa, "gap_pct": gap_pct,
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
