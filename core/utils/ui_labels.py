"""Static labels and tooltips for KPI visualisations."""

from __future__ import annotations

from typing import Dict


class _SafeLookup(dict):
    """Dictionary that returns the key when missing instead of raising ``KeyError``."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return str(key)


LABELS: Dict[str, str] = _SafeLookup(
    {
        "dpp_emision_pago": "Días Promedio a Pago (DPP)",
        "dic_emision_contab": "Días facturas hasta contabilizarse (DFC)",
        "dcp_contab_pago": "Días desde Contabilización al Pago (DCP)",
        "total_facturado": "Monto total facturado",
        "total_pagado_real": "Monto total pagado (real)",
    }
)

TOOLTIPS: Dict[str, str] = _SafeLookup(
    {
        "dpp_emision_pago": "Promedio de días entre emisión de factura y pago efectivo.",
        "dic_emision_contab": "Promedio de días entre que una factura pase a ser contabilizada.",
        "dcp_contab_pago": "Promedio de días entre contabilización y pago.",
        "total_facturado": "Incluye todo el monto facturado del periodo, incluso facturas aún sin pago.",
        "total_pagado_real": "Suma `monto_pagado_real`: usa `monto_autorizado` de la hoja Facturas cuando hay fecha de pago.",
        "desglose_facturado": "Comparativo del monto facturado entre documentos pagados y pendientes.",
    }
)

__all__ = ["LABELS", "TOOLTIPS"]
