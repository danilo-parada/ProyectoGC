"""Static labels and tooltips for KPI visualisations."""

from __future__ import annotations

from typing import Dict

LABELS: Dict[str, str] = {
    "dpp_emision_pago": "Días Promedio a Pago (DPP)",
    "dic_emision_contab": "Días a Ingreso Contable (DIC)",
    "dcp_contab_pago": "Días desde Contabilización al Pago (DCP)",
}

TOOLTIPS: Dict[str, str] = {
    "dpp_emision_pago": "Promedio de días entre emisión de factura y pago efectivo.",
    "dic_emision_contab": "Promedio de días entre emisión y registro contable.",
    "dcp_contab_pago": "Promedio de días entre contabilización y pago.",
    "total_facturado": "Incluye todo el monto facturado del periodo, incluso facturas aún sin pago.",
    "total_pagado": (
        "Monto cubierto considerando pagos realizados y montos con cuenta especial (CE); "
        "se contabiliza el monto CE cuando aplica."
    ),
}

__all__ = ["LABELS", "TOOLTIPS"]
