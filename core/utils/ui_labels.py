"""Static labels and tooltips for KPI visualisations."""

from __future__ import annotations

from typing import Dict

LABELS: Dict[str, str] = {
    "dpp_emision_pago": "Días Promedio a Pago (DPP)",
    "dic_emision_contab": "Días a Ingreso Contable (DIC)",
    "dcp_contab_pago": "Días desde Contabilización al Pago (DCP)",
    "brecha_porcentaje": "Brecha % (facturado vs pagado)",
}

TOOLTIPS: Dict[str, str] = {
    "dpp_emision_pago": "Promedio de días entre emisión de factura y pago efectivo.",
    "dic_emision_contab": "Promedio de días entre emisión y registro contable.",
    "dcp_contab_pago": "Promedio de días entre contabilización y pago.",
    "brecha_porcentaje": "Diferencia porcentual entre monto facturado (pagado) y monto pagado.",
}

__all__ = ["LABELS", "TOOLTIPS"]
